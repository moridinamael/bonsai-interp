"""Steering Bonsai with bundles of labelled poles: the neuron map as a control panel.

A knob is a bundle: the assignment-grade poles whose trigger is a word of one concept. A knob at c adds c x (pole max) to each
of its poles at every position. Outputs are measured on greedy continuations of neutral prompts, so they have no gradient.
"""
import collections

import numpy as np
import torch

from bonsai_interp.env import DEVICE

N = 12288
CONCEPTS = {
    "sports": "game team player players coach season league match goal score football basketball baseball soccer tennis golf stadium championship tournament athlete",
    "food": "food cook cooking recipe chicken cheese bread sauce meat soup dinner restaurant chef butter flour oven salad rice pasta sugar",
    "law": "law court judge lawyer attorney legal trial lawsuit statute contract plaintiff defendant jury justice appeal ruling rights attorneys courts verdict",
    "medicine": "doctor patient patients hospital disease treatment surgery cancer nurse medical clinic drug symptoms diagnosis therapy infection medicine vaccine blood health",
    "music": "music song songs band album guitar piano singer concert musician drums jazz melody lyrics orchestra violin rock tune choir singing",
    "weather": "weather rain snow storm wind cloud clouds temperature forecast sunny winter summer hurricane flood thunder ice fog drought humidity",
    "war": "war army soldier soldiers military battle troops enemy weapons navy officer general attack bomb missile tank invasion regiment combat guns",
    "money": "money bank price stock market investor investors profit tax loan dollars budget revenue finance economy wealth cash debt shares trading",
    "religion": "church god prayer priest faith temple bible holy religious bishop mosque spiritual worship christian muslim monk sacred saint heaven",
    "space": "space planet star stars moon galaxy orbit telescope astronaut rocket nasa mars solar universe comet asteroid satellite cosmic spacecraft",
    "software": "software code computer server programming database algorithm developer app python javascript function file network internet website bug data install",
    "animals": "dog cat horse bird fish animal animals wolf lion bear cattle sheep rabbit snake deer elephant tiger pet dogs",
    "vehicles": "car truck engine driver vehicle road highway tire train bus motorcycle fuel brake garage traffic wheels cars speed",
    "politics": "president election vote voters senator congress government minister party parliament campaign governor democrat republican policy mayor candidate political senate",
    "school": "school student students teacher class university college education classroom homework exam professor campus grade degree lecture tuition learning",
    "sea": "sea ocean ship boat sailor harbor coast waves captain island beach fishing vessel sailing tide port shore crew",
    "farming": "farm farmer crop crops soil garden seed harvest plants tree trees flower wheat corn field grass fertilizer orchard",
    "family": "mother father son daughter wife husband brother sister parents children family baby grandmother uncle aunt wedding marriage love",
    "film": "movie film actor actress director scene hollywood cinema television episode show character screen drama comedy theater audience",
    "crime": "police crime murder arrest suspect prison robbery detective thief gun victim killed officer sheriff gang jail stolen",
    "chemistry": "chemical acid molecule atom carbon oxygen hydrogen reaction compound protein enzyme metal electron nitrogen solution laboratory",
    "travel": "travel hotel flight airport tourist trip vacation passport journey city country mountain river village map tour",
    "fashion": "dress shirt shoes clothes fashion jacket hat coat wearing wore pants jeans skirt suit leather silk",
    "math": "number equation theorem proof matrix integral geometry algebra angle triangle sum formula variable probability calculus prime",
}


def concept_tokens(tok):
    """{concept: token ids of its words, spaced or not, lower or capitalised}."""
    out = {}
    for c, words in CONCEPTS.items():
        s = set()
        for w in words.split():
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                i = tok(v, add_special_tokens=False).input_ids
                if len(i) == 1:
                    s.add(i[0])
        out[c] = sorted(s)
    return out


def bundles(tok, assigned):
    """{concept: [(L, n, sign)]}: assignment-grade poles whose commonest trigger token is a word of the concept."""
    from bonsai_interp import atlas
    toks = concept_tokens(tok)
    word = {t: c for c, s in toks.items() for t in s}
    out = collections.defaultdict(list)
    for L in range(36):
        t = atlas.load(L)
        if t is None:
            continue
        d = t.select(["neuron", "sign", "hit_tok", "hit_trigger"]).to_pydict()
        for n, sg, ht, tr in zip(d["neuron"], d["sign"], d["hit_tok"], d["hit_trigger"]):
            if (L, n, sg) not in assigned or not ht:
                continue
            hits = [word.get(int(k)) for k, f in zip(ht, tr) if f]
            hits = [h for h in hits if h]
            if hits:
                out[collections.Counter(hits).most_common(1)[0][0]].append((L, n, sg))
    return dict(out), toks


class Panel:
    """The knobs of a set of bundles: V[L] is (knobs, N), what one unit of each knob adds at layer L."""

    def __init__(self, model, rec, pole_max, bundle_list):
        self.model, self.rec, self.k = model, rec, len(bundle_list)
        V = {}
        for j, poles in enumerate(bundle_list):
            for L, n, sg in poles:
                v = V.setdefault(L, np.zeros((self.k, N), np.float32))
                p = 0 if sg == "+" else 1
                v[j, n] += (1 if sg == "+" else -1) * pole_max[L, p, n]
        self.V = {L: torch.from_numpy(v).to(DEVICE) for L, v in V.items()}

    def install(self, theta):
        """theta: (rows, knobs). Row i of the batch gets theta[i]."""
        t = torch.as_tensor(theta, device=DEVICE, dtype=torch.float32)
        add = {L: t @ v for L, v in self.V.items()}
        self.rec.interventions = {L: (lambda a, m=m: a + m[:, None, :].to(a.dtype)) for L, m in add.items()}

    def clear(self):
        self.rec.interventions = {}


@torch.no_grad()
def generate(panel, tok, prompts, theta, new=32):
    """Greedy continuations of every prompt under every design. theta: (designs, knobs).
    Returns the encoding, ids (designs, prompts, new) and the steered log-probs (rows, new, vocab)."""
    tok.padding_side = "left"
    e = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
    D, P = len(theta), len(prompts)
    panel.install(np.repeat(np.asarray(theta, np.float32), P, axis=0))
    try:
        out = panel.model.generate(input_ids=e.input_ids.repeat(D, 1), attention_mask=e.attention_mask.repeat(D, 1),
                                   max_new_tokens=new, min_new_tokens=new, do_sample=False, pad_token_id=tok.pad_token_id,
                                   return_dict_in_generate=True, output_scores=True)
    finally:
        panel.clear()
        tok.padding_side = "right"
    gen = out.sequences[:, e.input_ids.shape[1]:]
    logp = torch.stack([torch.log_softmax(s.float(), -1).cpu() for s in out.scores], 1)
    return e, gen.view(D, P, -1), logp


@torch.no_grad()
def clean_logp(panel, e, gen):
    """Mean log-probability of the generated tokens under the unsteered model, per row: how natural the text is."""
    D, P, T = gen.shape
    g = gen.view(D * P, T)
    x = torch.cat([e.input_ids.repeat(D, 1), g], 1)
    m = torch.cat([e.attention_mask.repeat(D, 1), torch.ones_like(g)], 1)
    out = []
    for i in range(0, len(x), 8):
        lg = panel.model(input_ids=x[i:i + 8], attention_mask=m[i:i + 8]).logits.float()
        lp = torch.log_softmax(lg[:, -T - 1:-1], -1)
        out.append(lp.gather(2, x[i:i + 8, -T:, None])[..., 0].mean(1))
    return torch.cat(out).view(D, P)
