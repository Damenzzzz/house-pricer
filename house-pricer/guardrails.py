import re
from dataclasses import dataclass, field

MAX_QUERY_CHARS = 1000


_INJECTION = [
    r"(игнорир\w*|забуд\w*|отмен\w*|не\s+обращай\s+внимания\s+на|не\s+следуй)\s+(\w+\s+){0,3}"
    r"(инструкци|правил|указани|промпт|ограничени)",
    r"(ignore|disregard|forget|override)\s+(\w+\s+){0,3}(instructions?|rules|prompt|guidelines)",
    r"\b(ты\s+теперь|отныне\s+ты|теперь\s+ты|представь,?\s+что\s+ты|притворись)\b",
    r"\b(you\s+are\s+now|from\s+now\s+on\s+you|pretend\s+(to\s+be|you\s+are)|act\s+as\s+(a|an)\b)",
    r"(покажи|выведи|напечатай|раскрой|повтори)\s+(\w+\s+){0,3}(системн\w*\s+)?(промпт|инструкци|prompt)",
    r"(reveal|print|show|repeat)\s+(\w+\s+){0,3}(system\s+)?(prompt|instructions)",
    r"(системн\w+\s+промпт|system\s+prompt)",
    r"\b(jailbreak|DAN\s+mode|developer\s+mode|режим\s+разработчика)\b",
    r"(<\s*/?\s*(system|assistant|user)\s*>|\[/?INST\]|<\|im_(start|end)\|>|^\s*(system|assistant)\s*:)",
]
_INJECTION_RE = [re.compile(p, re.I | re.M) for p in _INJECTION]


_PII = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "телефон": r"(?<!\d)(\+7|8)[\s\-()]*7\d{2}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)",
    "карта": r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)",
    "ИИН": r"(?<!\d)\d{12}(?!\d)",
}
_PII_RE = {k: re.compile(p) for k, p in _PII.items()}


_OTHER_CITIES = r"(астан|нур-султан|шымкент|караганд|актоб|тараз|павлодар|усть-каменогорск|семе[йя]|" \
                r"атырау|костана|кызылорд|уральск|петропавловск|актау|туркестан|талдыкорган|конаев|" \
                r"капчага|москв|бишкек|ташкент)"
_SALE = r"\b(купить|куплю|покупк\w*|продаж\w*|продаю|ипотек\w*|в\s+собственность)\b"


@dataclass
class InputCheck:
    text: str
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pii: list[str] = field(default_factory=list)


def detect_injection(text: str) -> list[str]:
    return [m.group(0).strip() for r in _INJECTION_RE if (m := r.search(text or ""))]


def mask_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for kind, r in _PII_RE.items():
        text, n = r.subn(f"[{kind} скрыт]", text)
        found += [kind] * n
    return text, found


def domain_warnings(text: str) -> list[str]:
    q = (text or "").lower()
    out = []
    if m := re.search(_OTHER_CITIES, q):
        out.append(f"в базе только Алматы и Каскелен, а в запросе упомянут другой город ({m.group(0)}…)")
    if re.search(_SALE, q):
        out.append("в базе только помесячная аренда — продажу и ипотеку мы не оцениваем")
    return out


def check_input(text: str) -> InputCheck:
    text = (text or "")[:MAX_QUERY_CHARS]
    masked, pii = mask_pii(text)
    inj = detect_injection(masked)
    return InputCheck(text=masked, blocked=bool(inj),
                      reasons=[f"похоже на prompt injection: «{s}»" for s in inj],
                      warnings=domain_warnings(masked), pii=pii)


RENT_MIN, RENT_MAX = 10_000, 50_000_000
TOLERANCE = 0.02
NBSP = " "
FORBIDDEN = [r"переплачива", r"переплат[аиуы]"]


def _amounts(text: str) -> list[float]:
    body = re.sub(r"\[[\d,\s]+\]", " ", text)
    nums = [float(n.replace(" ", "").replace(NBSP, "").replace(" ", ""))
            for n in re.findall(r"\d[\d\s" + NBSP + " " + r"]{3,}", body)]
    return [n for n in nums if RENT_MIN <= n <= RENT_MAX]


def check_output(text: str, allowed_numbers, allowed_ids) -> dict:
    allowed_numbers = [float(a) for a in allowed_numbers if a is not None]
    found = _amounts(text)
    invented = [n for n in found
                if not any(abs(n - a) <= max(TOLERANCE * a, 1000) for a in allowed_numbers)]
    ids = set(re.findall(r"\[(\d+)\]", text))
    bad_ids = sorted(ids - {str(i) for i in allowed_ids})
    phrases = [p for p in FORBIDDEN if re.search(p, text, re.I)]
    cities = sorted({m.group(0) for m in re.finditer(_OTHER_CITIES, text.lower())})
    return {"ok": not (invented or bad_ids or phrases or cities), "numbers": len(found),
            "invented_numbers": invented, "invented_ids": bad_ids, "forbidden": phrases,
            "other_cities": cities}


def guard_output(text: str, allowed_numbers, allowed_ids, fallback: str) -> tuple[str, dict]:
    report = check_output(text, allowed_numbers, allowed_ids)
    return (text if report["ok"] else fallback), report
