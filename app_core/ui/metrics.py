import re
import unicodedata


# -------------------------
# NORMALISATION TEXTE
# -------------------------
def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# -------------------------
# EXTRACTION CONFIANCE OCR
# -------------------------
def extract_confidences(ocr_data: dict) -> list[float]:
    confidences = []
    for page in ocr_data.get("pages", []):
        for line in page.get("lines", []):
            for word in line.get("words", []):
                confidence = word.get("confidence")
                if confidence is not None:
                    confidences.append(float(confidence))
    return confidences


# -------------------------
# SCORE LEXICAL FRANÇAIS
# Mesure le % de mots reconnus comme du français réel,
# indépendamment du score de confiance interne du moteur.
# Utilise pyspellchecker si disponible, sinon un dictionnaire
# embarqué de ~3000 mots français courants comme fallback.
# -------------------------

# Dictionnaire de fallback — mots très fréquents dans les actes d'état civil
_FALLBACK_FR_WORDS = {
    "le","la","les","de","du","des","un","une","et","en","a","au","aux",
    "par","sur","sous","dans","avec","pour","que","qui","ou","ne","se",
    "il","elle","ils","elles","nous","vous","on","je","tu","me","te",
    "son","sa","ses","leur","leurs","mon","ma","mes","ton","ta","tes",
    "ce","cet","cette","ces","l","d","s","j","n","y","m",
    # actes civils
    "janvier","fevrier","mars","avril","mai","juin","juillet","aout",
    "septembre","octobre","novembre","decembre",
    "mil","neuf","cent","huit","sept","six","cinq","quatre","trois","deux","un",
    "trente","vingt","dix","onze","douze","treize","quatorze","quinze","seize",
    "ans","an","heures","heure","minutes","minute",
    "mariage","contrat","epoux","epouse","epoux","futur","futurs",
    "profession","sans","domicilie","domiciliee","domicilies",
    "paris","seine","marne","gers","orne","calvados",
    "fils","fille","pere","mere","veuf","veuve","decede","decedee","decedes",
    "officier","etat","civil","arrondissement","premier","deuxieme","troisieme",
    "chevalier","legion","honneur","medaille","militaire","croix","guerre",
    "notaire","contrat","recu","signe","lecture","faite","temoins","majeurs",
    "ont","ete","fait","sont","unis","loi","nom","prononcer","prononcu",
    "declare","declares","declarent","vouloir","prendre","autre","apres",
    "presence","rue","avenue","boulevard","place","impasse",
    "employe","employé","banque","commerce","cultivateur","menuisier",
    "comptable","demenageur","representant","marchand","marchande",
    "saisons","quatre","inspecteur","police","chauffeur",
    "devant","comparu","publiquement","maison","commune","commun",
    "partie","part","autre","une","ainsi","ledit","ladite",
}


def _load_spellchecker():
    """Charge SpellChecker (pyspellchecker). Retourne None si non installé."""
    try:
        from spellchecker import SpellChecker
        return SpellChecker(language="fr")
    except Exception:
        return None


# Instance unique chargée une seule fois
_SPELL = _load_spellchecker()
_SPELLCHECKER_AVAILABLE = _SPELL is not None


def compute_lexical_score(raw_text: str) -> dict:
    """
    Calcule le score lexical français du texte OCR.

    Retourne :
        lexical_score       : float 0-1  — ratio de vrais mots français
        lexical_pct         : float 0-100
        real_words          : int
        total_checked       : int
        spellchecker_used   : bool — True = pyspellchecker, False = fallback embarqué
    """
    words = normalize(raw_text).split()
    # On ignore les tokens trop courts (1-2 chars) et les nombres purs
    tokens = [w for w in words if len(w) >= 3 and not w.isdigit()]

    if not tokens:
        return {
            "lexical_score": 0.0,
            "lexical_pct": 0.0,
            "real_words": 0,
            "total_checked": 0,
            "spellchecker_used": _SPELLCHECKER_AVAILABLE,
        }

    if _SPELLCHECKER_AVAILABLE:
        # pyspellchecker : unknown() retourne les mots NON reconnus
        unknown = _SPELL.unknown(tokens)
        real_words = len(tokens) - len(unknown)
    else:
        # Fallback : dictionnaire embarqué
        real_words = sum(1 for w in tokens if w in _FALLBACK_FR_WORDS)

    score = real_words / len(tokens)

    return {
        "lexical_score": round(score, 4),
        "lexical_pct": round(score * 100, 2),
        "real_words": real_words,
        "total_checked": len(tokens),
        "spellchecker_used": _SPELLCHECKER_AVAILABLE,
    }


# -------------------------
# MÉTRIQUES OCR (heuristiques)
# -------------------------
def compute_ocr_quality_metrics(
    ocr_data: dict,
    word_count: int,
    doubtful_ratio: float,
    unique_words: int,
    avg_conf: float | None,
) -> dict:

    confidences = extract_confidences(ocr_data)

    density_score = round(unique_words / word_count * 100, 2) if word_count else 0.0

    high_conf_ratio = (
        round(sum(1 for c in confidences if c >= 0.8) / len(confidences) * 100, 2)
        if confidences else 0.0
    )

    avg_conf_norm = avg_conf if avg_conf is not None else 0.0

    # Score lexical — indépendant de la confiance interne du moteur
    raw_text = ocr_data.get("text", {}).get("raw", "")
    lex = compute_lexical_score(raw_text)
    lexical_score = lex["lexical_score"]

    # Score composite mis à jour :
    # - 25% confiance interne moteur  (biaisée mais informative)
    # - 35% score lexical français    (nouvelle métrique fiable)
    # - 25% ratio mots non-douteux    (seuil utilisateur)
    # - 15% densité lexicale          (diversité vocabulaire)
    composite_score = round(
        avg_conf_norm   * 0.25 +
        lexical_score   * 0.35 +
        (1 - doubtful_ratio / 100) * 0.25 +
        (density_score / 100) * 0.15,
        3,
    )

    return {
        "density_score": density_score,
        "high_conf_ratio": high_conf_ratio,
        "lexical_score": lex["lexical_pct"],          # % affiché dans l'UI
        "lexical_real_words": lex["real_words"],
        "lexical_total_checked": lex["total_checked"],
        "spellchecker_used": lex["spellchecker_used"],
        "composite_score": composite_score,
    }


# -------------------------
# CER
# -------------------------
def compute_cer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    hyp = normalize(hypothesis)

    if not ref:
        return 1.0

    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            new_dp[j] = min(
                new_dp[j - 1] + 1,
                dp[j] + 1,
                dp[j - 1] + cost,
            )
        dp = new_dp

    return dp[m] / max(n, m, 1)


# -------------------------
# WER
# -------------------------
def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize(reference).split()
    hyp_words = normalize(hypothesis).split()

    if not ref_words:
        return 1.0

    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            new_dp[j] = min(
                new_dp[j - 1] + 1,
                dp[j] + 1,
                dp[j - 1] + cost,
            )
        dp = new_dp

    return dp[m] / max(n, m, 1)