from stancedetection.data import loaders as data_loaders

TASK_MAPPINGS = {
    # Stance Benchmark
    "arc": {
        "task_dir": "ARC",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "unrelated", 1: "discuss", 2: "agree", 3: "disagree"},
    },
    "argmin": {
        "task_dir": "ArgMin",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "Argument_against", 1: "Argument_for"},
    },
    "fnc1": {
        "task_dir": "FNC-1",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "unrelated", 1: "discuss", 2: "agree", 3: "disagree"},
    },
    "iac1": {
        "task_dir": "IAC",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "anti", 1: "pro", 2: "other"},
    },
    "ibmcs": {
        "task_dir": "IBM_CLAIM_STANCE",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "CON", 1: "PRO"},
    },
    "perspectrum": {
        "task_dir": "PERSPECTRUM",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "UNDERMINE", 1: "SUPPORT"},
    },
    "scd": {
        "task_dir": "SCD",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "against", 1: "for"},
    },
    "semeval2016t6": {
        "task_dir": "SemEval2016Task6",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "AGAINST", 1: "FAVOR", 2: "NONE"},
    },
    "semeval2019t7": {
        "task_dir": "SemEval2019Task7",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "support", 1: "deny", 2: "query", 3: "comment"},
    },
    "snopes": {
        "task_dir": "Snopes",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "refute", 1: "agree"},
    },
    # Others
    "covidlies": {
        "task_dir": "covidlies",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "positive", 1: "negative"},
    },
    "emergent": {
        "task_dir": "emergent",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "against", 1: "for", 2: "observing"},
    },
    "mtsd": {
        "task_dir": "mtsd",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "AGAINST", 1: "FAVOR", 2: "NONE"},
    },
    "poldeb": {
        "task_dir": "politicalDebates",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "for", 1: "against"},
    },
    "rumor": {
        "task_dir": "rumor",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "endorse", 1: "deny", 2: "question", 3: "neutral", 4: "unrelated"},
    },
    "vast": {
        "task_dir": "vast",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "con", 1: "pro", 2: "neutral"},
    },
    "wtwt": {
        "task_dir": "wtwt",
        "loader": data_loaders.StanceLoader,
        "id2label": {0: "comment", 1: "refute", 2: "support", 3: "unrelated"},
    },
}

DOMAIN_TASK_MAPPINGS = {
    "news": [
        "fnc1",
        "fnc1-ours",
        "emergent",
        "snopes",
    ],
    "social_media": [
        "semeval2016t6",
        "semeval2019t7",
        "covidlies",
        "mtsd",
        "rumor",
        "wtwt",
    ],
    "debates": [
        "iac1",
        "arc",
        "poldeb",
        "perspectrum",
        "scd",
    ],
    "various": [
        "argmin",
        "vast",
        "ibmcs",
    ],
}

POSITIVE_LABELS = {
    "arc__agree",
    "argmin__Argument_for",
    "fnc1__agree",
    "iac1__pro",
    "perspectrum__SUPPORT",
    "scd__for",
    "semeval2016t6__FAVOR",
    "semeval2019t7__support",
    "snopes__agree",
    "emergent__for",
    "mtsd__FAVOR",
    "poldeb__for",
    "vast__pro",
    "wtwt__support",
    "rumor__endorse",
    "covidlies__positive",
}

NEGATIVE_LABELS = {
    "arc__disagree",
    "argmin__Argument_against",
    "fnc1__disagree",
    "iac1__anti",
    "ibmcs__CON",
    "perspectrum__UNDERMINE",
    "scd__against",
    "semeval2016t6__AGAINST",
    "semeval2019t7__deny",
    "snopes__refute",
    "emergent__against",
    "mtsd__AGAINST",
    "poldeb__against",
    "vast__con",
    "wtwt__refute",
    "rumor__deny",
    "covidlies__negative",
}

DISCUSS_LABELS = {
    "arc__discuss",
    "fnc1__discuss",
    "semeval2019t7__query",
    "emergent__observing",
    "wtwt__comment",
    "rumor__question",
}

OTHER_LABELS = {
    "arc__unrelated",
    "fnc1__unrelated",
    "iac1__other",
    "semeval2019t7__comment",
    "mtsd__NONE",
    "wtwt__unrelated",
    "rumor__unrelated",
}

NEUTRAL_LABELS = {
    "rumor__neutral",
    "vast__neutral",
}

RELATED_TASK_MAP = {
    "arc__agree": POSITIVE_LABELS,
    "arc__disagree": NEGATIVE_LABELS,
    "arc__discuss": DISCUSS_LABELS,
    "arc__unrelated": OTHER_LABELS,
    "argmin__Argument_against": NEGATIVE_LABELS,
    "argmin__Argument_for": POSITIVE_LABELS,
    "emergent__against": NEGATIVE_LABELS,
    "emergent__for": POSITIVE_LABELS,
    "emergent__observing": DISCUSS_LABELS,
    "fnc1__agree": POSITIVE_LABELS,
    "fnc1__disagree": NEGATIVE_LABELS,
    "fnc1__discuss": DISCUSS_LABELS,
    "fnc1__unrelated": OTHER_LABELS,
    "iac1__anti": NEGATIVE_LABELS,
    "iac1__other": OTHER_LABELS,
    "iac1__pro": POSITIVE_LABELS,
    "ibmcs__CON": NEGATIVE_LABELS,
    "ibmcs__PRO": POSITIVE_LABELS,
    "mtsd__AGAINST": NEGATIVE_LABELS,
    "mtsd__FAVOR": POSITIVE_LABELS,
    "mtsd__NONE": OTHER_LABELS,
    "perspectrum__SUPPORT": POSITIVE_LABELS,
    "perspectrum__UNDERMINE": NEGATIVE_LABELS,
    "poldeb__against": NEGATIVE_LABELS,
    "poldeb__for": POSITIVE_LABELS,
    "rumor__deny": NEGATIVE_LABELS,
    "rumor__endorse": POSITIVE_LABELS,
    "rumor__neutral": NEUTRAL_LABELS,
    "rumor__question": DISCUSS_LABELS,
    "rumor__unrelated": OTHER_LABELS,
    "scd__against": NEGATIVE_LABELS,
    "scd__for": POSITIVE_LABELS,
    "semeval2016t6__AGAINST": NEGATIVE_LABELS,
    "semeval2016t6__FAVOR": POSITIVE_LABELS,
    "semeval2016t6__NONE": OTHER_LABELS,
    "semeval2019t7__comment": OTHER_LABELS,
    "semeval2019t7__deny": NEGATIVE_LABELS,
    "semeval2019t7__query": DISCUSS_LABELS,
    "semeval2019t7__support": POSITIVE_LABELS,
    "snopes__agree": POSITIVE_LABELS,
    "snopes__refute": NEGATIVE_LABELS,
    "vast__con": NEGATIVE_LABELS,
    "vast__neutral": NEUTRAL_LABELS,
    "vast__pro": POSITIVE_LABELS,
    "wtwt__comment": DISCUSS_LABELS,
    "wtwt__refute": NEGATIVE_LABELS,
    "wtwt__support": POSITIVE_LABELS,
    "wtwt__unrelated": OTHER_LABELS,
    "covidlies__positive": POSITIVE_LABELS,
    "covidlies__negative": NEGATIVE_LABELS,
}
