/*  ═══════════════════════════════════════════════════════════════════════════
    Mise en Code — Import Pro Backend  v2
    ─────────────────────────────────────────────────────────────────────────
    POST /api/import-pro/parse
    
    A high-fidelity cookbook → algorithm-graph parser for Mise en Code.
    
    Improvements over v1
    --------------------
    1. STRUCTURED OUTPUT VIA TOOL USE
       Claude must call a single tool whose input_schema mirrors the recipe
       shape exactly. No more brittle markdown-fence JSON parsing.
    
    2. EXPLICIT DECOMPOSITION DOCTRINE
       The system prompt teaches a 5-stage pipeline (tokenise → map → place →
       lane → connect) plus thick rule tables:  verb → process,  modifier →
       knife cut,  heat phrase → enum,  appliance → tool.opts shape, etc.
    
    3. THE "FINE CHOPPED ONIONS" RULE
       Ingredient lines and instruction sentences are split into
       (noun  +  process verbs  +  modifier adjectives).  The noun is the
       ingredient; the verbs become processes; the adjectives become tool
       opts (knife cut, heat level, etc.).  No more "Fine Chopped Onion" as
       an ingredient name.
    
    4. processParams vs tool.opts DISCRIMINATOR
       Oven / Thermomix / KitchenAid / Sous-Vide / Blow-Torch / Fridge /
       Freezer  →  time & temp live in tool.opts.
       Stovetop pans, pots, wok, knife, manual tools, no-tool steps  →
       time & temp live in processParams (because tool.opts only has heat).
    
    5. SERVER-SIDE UID & VALIDATION PASS
       UIDs are generated server-side after parsing.  Grids are made
       rectangular.  connectsTo references are validated and dropped if
       dangling.  Flavour profile is estimated from the actual ingredients.
    
    6. EXTENDED THINKING
       Claude reasons in a thinking block before emitting the tool call.
       Costs a bit more, dramatically improves grid layout quality.
    
    Drop-in replacement for the previous server.js.  Same request shape.
    ═════════════════════════════════════════════════════════════════════════ */

import { config } from "dotenv";
config();

import express from "express";
import cors    from "cors";
import Anthropic from "@anthropic-ai/sdk";
import { z }  from "zod";

/*  ─── env / constants ─── */
const PORT          = process.env.PORT || 3001;
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || "*").split(",").map(s => s.trim());
const MODEL         = process.env.IMPORT_PRO_MODEL || "claude-sonnet-4-6";
const MAX_TOKENS    = Number(process.env.IMPORT_PRO_MAX_TOKENS || 12000);
const THINKING_BUDGET = Number(process.env.IMPORT_PRO_THINKING_BUDGET || 6000);
const ENABLE_THINKING = (process.env.IMPORT_PRO_THINKING ?? "1") !== "0";

const anthropic = new Anthropic();

/*  ═══════════════════════════════════════════════════════════════════════════
    SECTION 1 — SCHEMA ENUMS  (mirror of the front-end constants)
    Everything the LLM is allowed to emit is constrained here so we can
    validate post-hoc and refuse malformed payloads.
    ═════════════════════════════════════════════════════════════════════════ */

const HEAT_OPTS    = ["Off","Low","Medium-Low","Medium","Medium-High","High"];
const KNIFE_CUTS   = ["Thick Cut","Thin Cut","Slice","Batonnet","Julienne","Fine Julienne",
                      "Chiffonade","Large Dice","Medium Dice","Small Dice","Brunoise","Fine Brunoise"];
const OVEN_MODES   = ["Conventional","Fan Forced","Grill","Fan Grill","Bottom Heat","Top & Bottom"];
const KA_ATTACH    = ["Flat Beater","Wire Whip","Dough Hook","Flex Edge Beater"];
const KA_SPEEDS    = ["Stir","1","2","3","4","5","6","7","8","9","10"];
const TM_SPEEDS    = ["Gentle Stir","1","2","3","4","5","6","7","8","9","10","Turbo","Dough"];
const BLENDER_SPEEDS = ["Low","Medium","High","Pulse","Ice Crush"];
const GRATER_GRADES  = ["Super Fine","Fine","Medium","Coarse"];
const SIEVE_TYPES    = ["Fine Sieve","Mesh Strainer","Drum Sieve"];
const TIME_UNITS     = ["sec","min","hr"];
const ALLOWED_UNITS  = ["g","kg","ml","l","L","tsp","tbsp","cup","ea","pinch","bunch",
                        "cloves","slice","slices","strip","strips","sprig","stalks",
                        "to taste","×"];
const FLAVOUR_AXES   = ["sweetness","richness","spiciness","acidity","umami"];
const RECIPE_CATS    = ["Entrees","Mains","Desserts","Drinks & Cocktails"];

/*  ═══════════════════════════════════════════════════════════════════════════
    SECTION 2 — SYNONYM & MAPPING TABLES
    These short tables are stitched into the system prompt to seed Claude
    with strong priors.  They are NOT exhaustive — they are exemplars of the
    *kind* of mapping the model should perform.
    ═════════════════════════════════════════════════════════════════════════ */

/* Ingredient synonyms — region/term variants that map to canonical names.
   The format below becomes a flat list inside the system prompt. */
const INGREDIENT_SYNONYMS = {
  /* US ↔ AU/UK ↔ generic */
  "yellow onion":      "Brown Onion",
  "white onion":       "Brown Onion",
  "spanish onion":     "Brown Onion",
  "scallions":         "Spring Onion",
  "scallion":          "Spring Onion",
  "green onion":       "Spring Onion",
  "bell pepper":       "Capsicum",
  "red pepper":        "Capsicum",
  "green pepper":      "Capsicum",
  "cilantro":          "Fresh Coriander",
  "coriander leaves":  "Fresh Coriander",
  "ground coriander":  "Ground Coriander",
  "ground beef":       "Beef Mince",
  "minced beef":       "Beef Mince",
  "ground pork":       "Pork Mince",
  "minced pork":       "Pork Mince",
  "ground lamb":       "Lamb Mince",
  "ground chicken":    "Chicken Mince",
  "ground turkey":     "Turkey Mince",
  "double cream":      "Heavy Cream",
  "whipping cream":    "Heavy Cream",
  "thickened cream":   "Heavy Cream",
  "single cream":      "Heavy Cream",  /* approximation */
  "heavy whipping cream": "Heavy Cream",
  "all-purpose flour": "Plain Flour",
  "all purpose flour": "Plain Flour",
  "ap flour":          "Plain Flour",
  "powdered sugar":    "Icing Sugar",
  "confectioners sugar":"Icing Sugar",
  "confectioner's sugar":"Icing Sugar",
  "superfine sugar":   "Caster Sugar",
  "granulated sugar":  "Caster Sugar",
  "baking soda":       "Bicarb Soda",
  "sodium bicarbonate":"Bicarb Soda",
  "kosher salt":       "Salt",
  "table salt":        "Salt",
  "flaky sea salt":    "Sea Salt Flakes",
  "maldon":            "Sea Salt Flakes",
  "evoo":              "Extra Virgin Olive Oil",
  "extra-virgin olive oil":"Extra Virgin Olive Oil",
  "courgette":         "Zucchini",
  "aubergine":         "Eggplant",
  "rocket":            "Spinach",  /* loose, but better than nothing — Claude can warn */
  "prawn":             "Prawns",
  "shrimp":            "Prawns",
  "shrimps":           "Prawns",
  "stick of butter":   "Butter",   /* Claude must convert: 1 stick ≈ 113g */
  "knob of butter":    "Butter",   /* ≈ 15-20g */
};

/* ─────────────────────────────────────────────────────────────────────────
   INGREDIENT_TRANSLATIONS  — non-English names → canonical English names.
   Keys are lowercase (accents stripped where common), values are the same
   canonical English strings used in INGREDIENT_SYNONYMS / the ingredient
   library.  Used in two places:
     1. The system prompt synonym block, so the LLM resolves names correctly.
     2. estimateFlavour() resolver, so the flavour engine works on foreign
        recipes even if the LLM translated a name slightly differently.
   Organised by language for maintainability.
   ──────────────────────────────────────────────────────────────────────── */
const INGREDIENT_TRANSLATIONS = {

  /* ── SPANISH ───────────────────────────────────────────────────────────── */
  /* meats / minces */
  "carne molida de vacuno":   "Beef Mince",
  "carne molida de res":      "Beef Mince",
  "carne picada de vacuno":   "Beef Mince",
  "carne picada de res":      "Beef Mince",
  "carne molida":             "Beef Mince",   /* generic — context usually beef */
  "carne picada":             "Beef Mince",
  "carne de vacuno":          "Beef",
  "carne de res":             "Beef",
  "vacuno":                   "Beef",
  "carne molida de cerdo":    "Pork Mince",
  "carne picada de cerdo":    "Pork Mince",
  "cerdo":                    "Pork",
  "carne de cerdo":           "Pork",
  "carne molida de cordero":  "Lamb Mince",
  "cordero":                  "Lamb",
  "carne de cordero":         "Lamb",
  "carne molida de pollo":    "Chicken Mince",
  "pollo":                    "Chicken",
  "pechuga de pollo":         "Chicken Breast",
  "muslo de pollo":           "Chicken Thigh",
  "costillas":                "Short Rib",
  "tocino":                   "Bacon",
  "panceta":                  "Pancetta",
  "jamon":                    "Ham",
  "jamón":                    "Ham",
  "chorizo":                  "Chorizo",
  "salchicha":                "Sausage",
  /* vegetables */
  "cebolla":                  "Brown Onion",
  "cebolla blanca":           "Brown Onion",
  "cebolla amarilla":         "Brown Onion",
  "cebolla morada":           "Red Onion",
  "cebolla roja":             "Red Onion",
  "cebolleta":                "Spring Onion",
  "cebolla de verdeo":        "Spring Onion",
  "ajo":                      "Garlic",
  "diente de ajo":            "Garlic",
  "dientes de ajo":           "Garlic",
  "tomate":                   "Tomato",
  "tomates":                  "Tomato",
  "tomate en lata":           "Canned Tomatoes",
  "tomates en lata":          "Canned Tomatoes",
  "tomates triturados":       "Canned Tomatoes",
  "pasta de tomate":          "Tomato Paste",
  "pure de tomate":           "Tomato Paste",
  "puré de tomate":           "Tomato Paste",
  "pimiento":                 "Capsicum",
  "pimiento rojo":            "Capsicum",
  "pimiento verde":           "Capsicum",
  "pimiento amarillo":        "Capsicum",
  "aji":                      "Capsicum",
  "ají":                      "Capsicum",
  "aji verde":                "Capsicum",
  "paprika":                  "Paprika",
  "pimentón":                 "Paprika",
  "pimenton":                 "Paprika",
  "zanahoria":                "Carrot",
  "papa":                     "Potato",
  "papas":                    "Potato",
  "patata":                   "Potato",
  "patatas":                  "Potato",
  "camote":                   "Sweet Potato",
  "batata":                   "Sweet Potato",
  "zapallo":                  "Pumpkin",
  "calabaza":                 "Pumpkin",
  "choclo":                   "Corn",
  "maiz":                     "Corn",
  "maíz":                     "Corn",
  "poroto":                   "Bean",
  "frijol":                   "Bean",
  "frijoles":                 "Bean",
  "porotos":                  "Bean",
  "lentejas":                 "Lentils",
  "garbanzos":                "Chickpeas",
  "espinaca":                 "Spinach",
  "espinacas":                "Spinach",
  "acelga":                   "Silverbeet",
  "lechuga":                  "Lettuce",
  "apio":                     "Celery",
  "puerro":                   "Leek",
  "champiñon":                "Mushrooms",
  "champiñones":              "Mushrooms",
  "hongos":                   "Mushrooms",
  "berenjena":                "Eggplant",
  "zapallito":                "Zucchini",
  "zucchini":                 "Zucchini",
  "brocoli":                  "Broccoli",
  "brócoli":                  "Broccoli",
  /* pantry / dairy */
  "harina":                   "Plain Flour",
  "harina de trigo":          "Plain Flour",
  "harina común":             "Plain Flour",
  "azucar":                   "Caster Sugar",
  "azúcar":                   "Caster Sugar",
  "azúcar rubia":             "Brown Sugar",
  "azucar rubia":             "Brown Sugar",
  "azúcar morena":            "Brown Sugar",
  "sal":                      "Salt",
  "sal fina":                 "Salt",
  "sal marina":               "Sea Salt Flakes",
  "pimienta":                 "Black Pepper",
  "pimienta negra":           "Black Pepper",
  "aceite de oliva":          "Olive Oil",
  "aceite de oliva extra virgen": "Extra Virgin Olive Oil",
  "aceite vegetal":           "Vegetable Oil",
  "aceite":                   "Vegetable Oil",
  "mantequilla":              "Butter",
  "manteca":                  "Butter",           /* AR/CL/PY */
  "crema":                    "Heavy Cream",
  "crema de leche":           "Heavy Cream",
  "nata":                     "Heavy Cream",      /* ES */
  "leche":                    "Milk",
  "leche entera":             "Milk",
  "huevo":                    "Egg",
  "huevos":                   "Egg",
  "queso":                    "Cheese",
  "queso parmesano":          "Parmesan",
  "queso rallado":            "Parmesan",
  "caldo":                    "Stock",
  "caldo de pollo":           "Chicken Stock",
  "caldo de carne":           "Beef Stock",
  "caldo de verduras":        "Vegetable Stock",
  "vino tinto":               "Red Wine",
  "vino blanco":              "White Wine",
  "vinagre":                  "White Vinegar",
  "vinagre de vino tinto":    "Red Wine Vinegar",
  "vinagre balsámico":        "Balsamic Vinegar",
  "vinagre balsamico":        "Balsamic Vinegar",
  /* herbs / spices */
  "perejil":                  "Parsley",
  "cilantro":                 "Fresh Coriander",
  "oregano":                  "Oregano",
  "orégano":                  "Oregano",
  "comino":                   "Cumin",
  "laurel":                   "Bay Leaf",
  "tomillo":                  "Thyme",
  "romero":                   "Rosemary",
  "albahaca":                 "Basil",
  "aji molido":               "Chilli Flakes",
  "ají molido":               "Chilli Flakes",
  "merkén":                   "Chilli Flakes",    /* CL smoked chilli */
  "merken":                   "Chilli Flakes",
  "limon":                    "Lemon",
  "limón":                    "Lemon",
  "lima":                     "Lime",
  /* ── FRENCH ────────────────────────────────────────────────────────────── */
  "beurre":                   "Butter",
  "farine":                   "Plain Flour",
  "sucre":                    "Caster Sugar",
  "sucre glace":              "Icing Sugar",
  "sel":                      "Salt",
  "poivre":                   "Black Pepper",
  "oignon":                   "Brown Onion",
  "oignons":                  "Brown Onion",
  "ail":                      "Garlic",
  "tomate":                   "Tomato",
  "crème fraîche":            "Sour Cream",
  "creme fraiche":            "Sour Cream",
  "crème":                    "Heavy Cream",
  "lait":                     "Milk",
  "oeuf":                     "Egg",
  "oeufs":                    "Egg",
  "viande hachée":            "Beef Mince",
  "boeuf haché":              "Beef Mince",
  "poulet":                   "Chicken",
  "porc":                     "Pork",
  "boeuf":                    "Beef",
  "agneau":                   "Lamb",
  "lardons":                  "Bacon",
  "huile d'olive":            "Olive Oil",
  "citron":                   "Lemon",
  "persil":                   "Parsley",
  "thym":                     "Thyme",
  "romarin":                  "Rosemary",
  /* ── ITALIAN ────────────────────────────────────────────────────────────── */
  "burro":                    "Butter",
  "farina":                   "Plain Flour",
  "zucchero":                 "Caster Sugar",
  "sale":                     "Salt",
  "pepe":                     "Black Pepper",
  "cipolla":                  "Brown Onion",
  "aglio":                    "Garlic",
  "pomodoro":                 "Tomato",
  "pomodori":                 "Tomato",
  "concentrato di pomodoro":  "Tomato Paste",
  "olio di oliva":            "Olive Oil",
  "olio extravergine":        "Extra Virgin Olive Oil",
  "uovo":                     "Egg",
  "uova":                     "Egg",
  "latte":                    "Milk",
  "panna":                    "Heavy Cream",
  "carne macinata":           "Beef Mince",
  "macinato di manzo":        "Beef Mince",
  "pollo":                    "Chicken",
  "maiale":                   "Pork",
  "manzo":                    "Beef",
  "agnello":                  "Lamb",
  "limone":                   "Lemon",
  "prezzemolo":               "Parsley",
  "basilico":                 "Basil",
  "timo":                     "Thyme",
  "rosmarino":                "Rosemary",
};

/* Verb → canonical process key.  Many cookbook verbs map to one process. */
const VERB_TO_PROCESS = {
  /* size & shape */
  "chop": "chop", "chopped": "chop", "chops": "chop",
  "dice": "dice", "diced": "dice",
  "mince": "mince", "minced": "mince",
  "slice": "slice", "sliced": "slice", "slicing": "slice",
  "julienne": "julienne", "julienned": "julienne",
  "grate": "grate", "grated": "grate", "grating": "grate",
  "shred": "shred", "shredded": "shred",
  "zest": "zest", "zested": "zest", "zesting": "zest",
  "crush": "crush", "crushed": "crush", "crushing": "crush",
  "grind": "grind", "ground": "grind", "grinding": "grind",
  /* prep */
  "peel": "peel", "peeled": "peel", "peeling": "peel",
  "trim": "trim", "trimmed": "trim",
  "wash": "wash", "washed": "wash", "rinse": "wash", "rinsed": "wash",
  "dry": "dry", "dried": "dry", "pat dry": "dry",
  "deseed": "de-seed", "de-seed": "de-seed", "seed": "de-seed",
  "shell": "shell", "shelled": "shell",
  "bone": "bone", "boned": "bone", "debone": "bone",
  "weigh": "weigh", "weighed": "weigh", "measure": "measure",
  /* hydration */
  "soak": "soak", "soaked": "soak",
  "marinate": "marinate", "marinated": "marinate", "marinade": "marinate",
  "cure": "cure", "cured": "cure",
  "brine": "brine", "brined": "brine",
  "steep": "steep", "steeped": "steep",
  "infuse": "infuse", "infused": "infuse", "infusing": "infuse",
  /* combining */
  "mix": "mix", "mixed": "mix", "mixing": "mix", "combine": "mix", "combined": "mix",
  "stir": "stir", "stirred": "stir", "stirring": "stir",
  "fold": "fold", "folded": "fold", "folding": "fold",
  "whisk": "whisk", "whisked": "whisk", "whisking": "whisk",
  "beat": "beat", "beaten": "beat", "beating": "beat",
  "toss": "toss", "tossed": "toss", "tossing": "toss",
  "cream": "cream", "creamed": "cream",
  "disperse": "disperse",
  "dissolve": "dissolve", "dissolved": "dissolve",
  "slurry": "slurry",
  /* emulsion / foam */
  "emulsify": "emulsify", "emulsified": "emulsify",
  "whip": "whip", "whipped": "whip", "whipping": "whip",
  "froth": "froth",
  "foam": "foam", "foamed": "foam",
  /* structure */
  "knead": "knead", "kneaded": "knead", "kneading": "knead",
  "laminate": "laminate", "laminated": "laminate",
  "rest": "rest", "rested": "rest", "resting": "rest",
  "ferment": "ferment", "fermented": "ferment",
  "proof": "proof", "proofed": "proof", "proofing": "proof",
  "thicken": "thicken", "thickened": "thicken",
  "set": "set",
  /* dry heat */
  "bake": "bake", "baked": "bake", "baking": "bake",
  "roast": "roast", "roasted": "roast", "roasting": "roast",
  "toast": "toast", "toasted": "toast", "toasting": "toast",
  "grill": "grill", "grilled": "grill", "grilling": "grill",
  "broil": "broil", "broiled": "broil",
  "sear": "sear", "seared": "sear", "searing": "sear",
  "saute": "sauté", "sauté": "sauté", "sautéed": "sauté", "sauteed": "sauté", "sauteing": "sauté",
  "pan-roast": "pan-roast", "pan roast": "pan-roast",
  "brown": "sear", "browned": "sear", /* "brown the meat" → sear */
  "caramelize": "reduce", "caramelise": "reduce", "caramelizing": "reduce",
  /* moist heat */
  "boil": "boil", "boiled": "boil", "boiling": "boil",
  "simmer": "simmer", "simmered": "simmer", "simmering": "simmer",
  "poach": "poach", "poached": "poach", "poaching": "poach",
  "steam": "steam", "steamed": "steam", "steaming": "steam",
  "blanch": "blanch", "blanched": "blanch", "blanching": "blanch",
  "parboil": "parboil", "parboiled": "parboil",
  /* fat-mediated */
  "shallow-fry": "shallow-fry", "shallow fry": "shallow-fry",
  "pan-fry": "pan-fry", "pan fry": "pan-fry", "pan-fried": "pan-fry",
  "deep-fry": "deep-fry", "deep fry": "deep-fry", "deep-fried": "deep-fry",
  "fry": "pan-fry", "fried": "pan-fry", "frying": "pan-fry",
  "confit": "confit",
  /* combination */
  "braise": "braise", "braised": "braise", "braising": "braise",
  "stew": "stew", "stewed": "stew",
  "pressure cook": "pressure cook", "pressure-cook": "pressure cook",
  "pot-roast": "pot-roast", "pot roast": "pot-roast",
  /* reduce */
  "reduce": "reduce", "reduced": "reduce", "reducing": "reduce",
  "evaporate": "evaporate",
  "concentrate": "concentrate", "concentrated": "concentrate",
  "deglaze": "deglaze", "deglazed": "deglaze",
  "glaze": "glaze", "glazed": "glaze",
  /* cool */
  "cool": "cool", "cooled": "cool", "cooling": "cool",
  "chill": "chill", "chilled": "chill", "chilling": "chill",
  "shock": "shock", "shocked": "shock",
  "refrigerate": "refrigerate", "refrigerated": "refrigerate",
  "freeze": "freeze", "frozen": "freeze",
  /* preserve */
  "pickle": "pickle", "pickled": "pickle",
  "smoke": "smoke", "smoked": "smoke",
  /* finish */
  "season": "season", "seasoned": "season",
  "garnish": "garnish", "garnished": "garnish",
  "dress": "dress", "dressed": "dress", "drizzle": "dress",
  /* assemble */
  "layer": "layer", "layered": "layer",
  "fill": "fill", "filled": "fill",
  "plate": "plate", "plated": "plate",
  "portion": "portion", "portioned": "portion", "divide": "portion",
  "carve": "carve", "carved": "carve",
  "sauce": "sauce", "sauced": "sauce",
  /* control */
  "taste": "taste",
  "check": "check doneness",
  "adjust": "adjust",
  "filter": "filter", "filtered": "filter",
  "strain": "strain", "strained": "strain", "straining": "strain",
  "drain": "strain", "drained": "strain", "draining": "strain",
  "sift": "sift", "sifted": "sift",
  "press": "press", "pressed": "press",
  "extract": "extract",
  "skim": "skim", "skimmed": "skim",
  "decant": "decant",
};

/* Modifier (adjective / adverb) → knife cut option.
   "fine chop" → cut="Brunoise";  "thin slice" → cut="Thin Cut".  */
const MOD_TO_KNIFE_CUT = {
  /* very fine */
  "fine brunoise":   "Fine Brunoise",
  "very fine":       "Brunoise",
  "extremely fine":  "Brunoise",
  "finely":          "Brunoise",   /* "finely chopped" / "finely diced" */
  "fine":            "Brunoise",
  "minced":          "Brunoise",
  /* small */
  "small dice":      "Small Dice",
  "small":           "Small Dice",
  /* medium */
  "medium dice":     "Medium Dice",
  "medium":          "Medium Dice",
  "diced":           "Medium Dice",
  "cubed":           "Medium Dice",
  "in cubes":        "Medium Dice",
  /* large */
  "large dice":      "Large Dice",
  "large":           "Large Dice",
  "rough":           "Large Dice",
  "roughly":         "Large Dice",
  "coarse":          "Large Dice",
  "coarsely":        "Large Dice",
  "chunks":          "Large Dice",
  /* baton */
  "batonnet":        "Batonnet",
  "batons":          "Batonnet",
  "sticks":          "Batonnet",
  "thick strips":    "Batonnet",
  /* julienne */
  "julienne":        "Julienne",
  "fine julienne":   "Fine Julienne",
  "matchsticks":     "Julienne",
  "matchstick":      "Julienne",
  "thin strips":     "Julienne",
  /* chiffonade */
  "chiffonade":      "Chiffonade",
  "ribbons":         "Chiffonade",
  /* slice */
  "thin slice":      "Thin Cut",
  "thinly sliced":   "Thin Cut",
  "thinly":          "Thin Cut",
  "thin":            "Thin Cut",
  "paper-thin":      "Thin Cut",
  "thick slice":     "Thick Cut",
  "thick":           "Thick Cut",
  "thickly":         "Thick Cut",
  "sliced":          "Slice",
};

/* Heat phrase → enum.  Simple lowercase exact / contains match. */
const HEAT_PHRASES = [
  /* check most-specific first */
  ["very low",        "Low"],
  ["low and slow",    "Low"],
  ["low heat",        "Low"],
  ["gentle heat",     "Low"],
  ["medium-low",      "Medium-Low"],
  ["medium low",      "Medium-Low"],
  ["med-low",         "Medium-Low"],
  ["moderate",        "Medium"],
  ["medium-high",     "Medium-High"],
  ["medium high",     "Medium-High"],
  ["med-high",        "Medium-High"],
  ["high heat",       "High"],
  ["high",            "High"],
  ["medium",          "Medium"],
  ["low",             "Low"],
];

/* Verb → tool-id heuristic.  When a verb implies a tool but none was
   mentioned, this is the fallback.  Some verbs imply *families* of tools;
   we pick a sensible default and let post-processing or a warning surface
   the choice. */
const VERB_TO_TOOL = {
  /* heat / cook */
  "bake":          "oven",
  "roast":         "oven",
  "broil":         "oven",
  "grill":         "oven",        /* unless context suggests outdoor BBQ */
  "toast":         "oven",
  "sear":          "cast-iron-pan",
  "sauté":         "cast-iron-pan",
  "pan-fry":       "cast-iron-pan",
  "shallow-fry":   "cast-iron-pan",
  "deep-fry":      "stock-pot",
  "stir-fry":      "wok",
  "boil":          "reg-pot",
  "simmer":        "saucepan",
  "poach":         "saucepan",
  "steam":         "stock-pot",
  "blanch":        "reg-pot",
  "parboil":       "reg-pot",
  "braise":        "cast-iron-pot",
  "stew":          "cast-iron-pot",
  "confit":        "cast-iron-pot",
  "reduce":        "saucepan",
  "deglaze":       "cast-iron-pan",
  /* prep / shape */
  "chop":          "knife",
  "dice":          "knife",
  "mince":         "knife",
  "slice":         "knife",
  "julienne":      "knife",
  "batonnet":      "knife",
  "chiffonade":    "knife",
  "grate":         "grater",
  "shred":         "grater",
  "zest":          "grater",
  "peel":          "knife",
  /* mix / combine */
  "whisk":         "whisk",       /* unless context says kitchenaid / thermomix */
  "beat":          "kitchenaid",
  "whip":          "kitchenaid",
  "knead":         "kitchenaid",
  "fold":          "spatula",
  "stir":          "spatula",
  "mix":           "bowl",
  /* cool */
  "chill":         "fridge",
  "refrigerate":   "fridge",
  "freeze":        "freezer",
  /* misc */
  "strain":        "sieve",
  "drain":         "colander",
  "sift":          "sieve",
  "filter":        "sieve",
  "weigh":         "scale",
  "measure":       "scale",
};

/* Common time phrase patterns — how Claude should parse:
   "for X minutes/seconds/hours"     → _time=X, _timeUnit=min/sec/hr
   "X-Y minutes" (range)             → _time=Y (upper bound), _timeUnit=min
   "X to Y minutes"                  → _time=Y, _timeUnit=min
   "until X" (no number)             → don't set _time
   These rules are explained in the system prompt. */

/*  ═══════════════════════════════════════════════════════════════════════════
    SECTION 3 — INGREDIENT FLAVOUR HEURISTICS  (server-side flavour estimator)
    Used purely as a sanity backstop if the LLM gives a flat 50/50/0/20/20.
    Keys are normalised lowercase ingredient names; values are 0–1 weights
    on the five axes (s, r, p, a, u) — the same convention the front-end
    uses for custom ingredients.
    ═════════════════════════════════════════════════════════════════════════ */
const FLAVOUR_HINTS = {
  /* sweets */
  "caster sugar":      { s:1.0, r:0.05, p:0,   a:0,    u:0   },
  "brown sugar":       { s:0.95,r:0.10, p:0,   a:0,    u:0   },
  "icing sugar":       { s:1.0, r:0,    p:0,   a:0,    u:0   },
  "honey":             { s:0.85,r:0.05, p:0,   a:0.10, u:0   },
  "maple syrup":       { s:0.85,r:0.10, p:0,   a:0.05, u:0   },
  "condensed milk":    { s:0.85,r:0.50, p:0,   a:0,    u:0   },
  /* fats / dairy */
  "butter":            { s:0.05,r:0.95, p:0,   a:0,    u:0.10},
  "heavy cream":       { s:0.10,r:0.85, p:0,   a:0,    u:0.05},
  "milk":              { s:0.10,r:0.30, p:0,   a:0,    u:0   },
  "olive oil":         { s:0,   r:0.70, p:0,   a:0.05, u:0   },
  "extra virgin olive oil":{s:0,r:0.75, p:0,   a:0.10, u:0   },
  /* heat / pepper */
  "chilli":            { s:0,   r:0,    p:0.95,a:0.10, u:0   },
  "red chilli":        { s:0,   r:0,    p:0.95,a:0.10, u:0   },
  "jalapeño":          { s:0,   r:0,    p:0.65,a:0.10, u:0   },
  "habanero":          { s:0,   r:0,    p:0.95,a:0.10, u:0   },
  "chilli flakes":     { s:0,   r:0,    p:0.85,a:0.05, u:0   },
  "black pepper":      { s:0,   r:0,    p:0.20,a:0,    u:0.05},
  "cayenne":           { s:0,   r:0,    p:0.80,a:0,    u:0   },
  "sriracha":          { s:0.10,r:0,    p:0.70,a:0.20, u:0.05},
  /* acid */
  "lemon":             { s:0.05,r:0,    p:0,   a:0.85, u:0   },
  "lemon juice":       { s:0,   r:0,    p:0,   a:0.95, u:0   },
  "lime":              { s:0.05,r:0,    p:0,   a:0.85, u:0   },
  "lime juice":        { s:0,   r:0,    p:0,   a:0.95, u:0   },
  "balsamic vinegar":  { s:0.20,r:0.05, p:0,   a:0.75, u:0.10},
  "white vinegar":     { s:0,   r:0,    p:0,   a:0.95, u:0   },
  "red wine vinegar":  { s:0.05,r:0,    p:0,   a:0.85, u:0.05},
  "apple cider vinegar":{s:0.05,r:0,    p:0,   a:0.85, u:0   },
  "white wine":        { s:0.10,r:0.05, p:0,   a:0.40, u:0   },
  "red wine":          { s:0.10,r:0.10, p:0,   a:0.40, u:0.05},
  /* umami */
  "parmesan":          { s:0,   r:0.50, p:0,   a:0.05, u:0.85},
  "pecorino romano":   { s:0,   r:0.55, p:0,   a:0.05, u:0.85},
  "soy sauce":         { s:0.05,r:0,    p:0,   a:0.05, u:0.95},
  "fish sauce":        { s:0.10,r:0,    p:0,   a:0.05, u:0.95},
  "miso paste":        { s:0.05,r:0.05, p:0,   a:0.05, u:0.90},
  "tomato paste":      { s:0.10,r:0,    p:0,   a:0.20, u:0.65},
  "anchovies":         { s:0,   r:0.10, p:0,   a:0.10, u:0.95},
  "worcestershire":    { s:0.10,r:0,    p:0,   a:0.20, u:0.70},
  "guanciale":         { s:0,   r:0.85, p:0,   a:0,    u:0.65},
  "pancetta":          { s:0,   r:0.80, p:0,   a:0,    u:0.65},
  "bacon":             { s:0.05,r:0.85, p:0,   a:0,    u:0.65},
  /* meats */
  "beef mince":        { s:0,   r:0.45, p:0,   a:0,    u:0.40},
  "beef chuck":        { s:0,   r:0.55, p:0,   a:0,    u:0.45},
  "short rib":         { s:0,   r:0.85, p:0,   a:0,    u:0.50},
  "chicken breast":    { s:0,   r:0.20, p:0,   a:0,    u:0.25},
  "chicken thigh":     { s:0,   r:0.40, p:0,   a:0,    u:0.30},
  "pork belly":        { s:0,   r:0.90, p:0,   a:0,    u:0.45},
  /* aromatics — light touches only */
  "garlic":            { s:0,   r:0,    p:0.05,a:0,    u:0.20},
  "brown onion":       { s:0.10,r:0,    p:0,   a:0,    u:0.10},
  "red onion":         { s:0.05,r:0,    p:0,   a:0.05, u:0.10},
};

/* (continued in part 2 — system prompt builder, tool schema, endpoint) */

/*  ═══════════════════════════════════════════════════════════════════════════
    SECTION 4 — LIBRARY SERIALISATION
    Transform the front-end's library payloads into compact text for the
    system prompt.  Names are essential (the LLM matches on them); we
    sacrifice full descriptions for token economy.
    ═════════════════════════════════════════════════════════════════════════ */

const serialiseIngredients = (lib = [], custom = {}) => {
  /* lib: [{id, n, cat}]   custom: { Category: [{id, n, u, isCust}] } */
  const lines = [];
  /* Group by category for the prompt — easier for Claude to scan */
  const byCat = {};
  for (const i of lib) {
    const c = i.cat || "Misc";
    (byCat[c] ??= []).push(`${i.id} → ${i.n}`);
  }
  for (const [cat, custList] of Object.entries(custom || {})) {
    for (const ci of custList) {
      (byCat[cat] ??= []).push(`${ci.id} → ${ci.n} [custom]`);
    }
  }
  for (const cat of Object.keys(byCat).sort()) {
    lines.push(`  • ${cat}:`);
    /* keep one ingredient per line for readability */
    for (const entry of byCat[cat]) lines.push(`      ${entry}`);
  }
  return lines.join("\n");
};

const serialiseTools = (lib = [], custom = {}) => {
  const lines = [];
  for (const t of lib) lines.push(`  ${t.id} → ${t.n}`);
  for (const [, custList] of Object.entries(custom || {})) {
    for (const ct of custList) lines.push(`  ${ct.id} → ${ct.n} [custom]`);
  }
  return lines.join("\n");
};

const serialiseProcesses = (lib = [], custom = {}) => {
  const allCustom = Object.values(custom || {}).flat();
  const all = [...lib, ...allCustom];
  /* group into rows of ~6 for compactness */
  const rows = [];
  for (let i = 0; i < all.length; i += 6) {
    rows.push("  " + all.slice(i, i + 6).join(", "));
  }
  return rows.join("\n");
};

const serialiseSynonyms = () => {
  const regional = Object.entries(INGREDIENT_SYNONYMS)
    .map(([k, v]) => `  "${k}" → "${v}"`).join("\n");
  const translations = Object.entries(INGREDIENT_TRANSLATIONS)
    .map(([k, v]) => `  "${k}" → "${v}"`).join("\n");
  return `Regional / variant spellings (English only):\n${regional}\n\nMultilingual translations (es/fr/it → canonical English):\n${translations}`;
};

/*  ═══════════════════════════════════════════════════════════════════════════
    SECTION 5 — THE SYSTEM PROMPT
    The heart of v2.  Deeply opinionated about *how* a cookbook recipe
    must be decomposed.  The prompt is large (~6k tokens) but heavily
    cached by Anthropic between calls because the library section
    rarely changes.
    ═════════════════════════════════════════════════════════════════════════ */

const buildSystemPrompt = (ctx) => `
You are IMPORT PRO, the recipe-structuring agent for Mise en Code — an app that
represents recipes as a *grid algorithm*: rows are time-steps, columns are
parallel work streams ("lanes"), and each non-empty cell is a STEP that
combines (a) ingredients used at that moment, (b) processes (verbs), (c) a
single tool with its options, and (d) optional connectsTo arrows when output
physically migrates between lanes.

Your only output is a single tool call to \`submit_recipe_draft\`.  Do NOT
write prose.  Do NOT write markdown.  The tool's input_schema is the contract.

═══════════════════════════════════════════════════════════════════════════
PART A — THE 5-STAGE DECOMPOSITION PIPELINE
═══════════════════════════════════════════════════════════════════════════

A cookbook recipe is *narrative*.  Mise en Code is a *graph*.  Your job is
the translation.  Walk these stages in order, in the thinking block, before
producing the final tool call.

╭─ Stage 1 ── Normalise the source ──────────────────────────────────────╮
│  • Detect language (en / es / fr / it / other).                        │
│                                                                        │
│  ★ MANDATORY TRANSLATION STEP (non-English recipes):                   │
│    If the source is NOT English, translate every ingredient core noun  │
│    to English in your thinking block BEFORE matching it to the library │
│    or synonym table.  Follow this sequence every time:                 │
│                                                                        │
│    1. Translate the core noun to plain English.                        │
│       e.g. "Carne Molida de Vacuno" → "beef mince"                    │
│            "Mantequilla"            → "butter"                         │
│            "Huevos"                 → "eggs"                           │
│            "Viande hachée"          → "beef mince"                     │
│            "Cipolla"                → "brown onion"                    │
│    2. Apply the synonym/translation table (Part C) to resolve the      │
│       canonical English library name.                                  │
│       e.g. "beef mince" → "Beef Mince"  (canonical library name)      │
│    3. The `name` field in the output MUST always be the canonical      │
│       English library name — NEVER the original foreign-language text. │
│       ❌ name: "Carne Molida de Vacuno"  (WRONG — raw foreign text)    │
│       ✓  name: "Beef Mince"             (RIGHT — canonical English)    │
│    4. Part C contains a Multilingual translations section for common   │
│       es / fr / it ingredient names.  Consult it first before trying  │
│       to translate from general knowledge.                             │
│    5. If a foreign ingredient truly has no close English equivalent    │
│       (very regional), translate as closely as possible and flag as a  │
│       candidate using the translated English name, not the original.   │
│                                                                        │
│  • Identify the structural sections: title, description, servings,     │
│    ingredient list(s), method/instructions, notes.                     │
│  • Separate ingredient *groups* (cookbook headings like "Para la       │
│    salsa:", "Marinada:", "Topping:") — each group is a strong hint    │
│    for a separate lane or sub-component.                               │
│  • Strip noise: serving suggestions, anecdotes, photo captions, ads.   │
╰────────────────────────────────────────────────────────────────────────╯

╭─ Stage 2 ── Decompose every INGREDIENT LINE ───────────────────────────╮
│  Cookbook lines look like:                                             │
│      "2 medium yellow onions, finely chopped"                          │
│      "200 g guanciale, cut into thick batons"                          │
│      "3 cloves of garlic, minced"                                      │
│      "1 cup heavy cream, warm"                                         │
│      "salt and pepper, to taste"                                       │
│                                                                        │
│  Decompose each line into SIX slots:                                  │
│    1.  QUANTITY        →  numeric (handle ranges → upper bound)        │
│    2.  UNIT            →  one of: g, kg, ml, l, tsp, tbsp, cup, ea,    │
│                            pinch, bunch, cloves, slice, strip, sprig,  │
│                            stalks, "to taste"                          │
│    3.  CORE NOUN       →  the ingredient itself.  Match a canonical    │
│                            id; never include verbs or modifiers.       │
│    4.  PRE-PROCESSES   →  any past-participle verbs ("chopped",        │
│                            "minced", "grated", "softened") that imply  │
│                            a process MUST be performed before this     │
│                            ingredient is used.                         │
│    5.  MODIFIERS       →  size/heat/state adjectives that map to TOOL  │
│                            OPTS, not new ingredients ("fine",          │
│                            "thinly", "warm", "room temperature").      │
│                                                                        │
│    6.  QUALIFIERS      →  quality/fat/spec descriptors that are NOT    │
│                            part of the core ingredient identity.       │
│                            e.g. "8% grasa", "extra lean", "sin hueso", │
│                            "bajo en grasa", "boneless", "skinless",    │
│                            "full-fat", "low-fat", "80/20".             │
│                            Strip these from `name` and append them to  │
│                            the step's `comment` field as a note.       │
│                            e.g. comment: "Beef Mince: 8% fat"          │
│                            NEVER let a qualifier bleed into `name`.    │
│                            ❌ name: "Carne Molida 8% grasa"            │
│                            ✓  name: "Beef Mince"                       │
│                               comment: "Beef Mince: 8% fat"           │
│  Examples:                                                             │
│    "2 medium yellow onions, finely chopped"                           │
│      qty=2  unit=ea  noun=Brown Onion (id=onion)                      │
│      pre-processes=[chop]  modifiers=[finely → cut=Brunoise]          │
│                                                                        │
│    "1 stick of butter, softened"                                      │
│      qty=113  unit=g  noun=Butter (id=butter)                         │
│      pre-processes=[]  modifiers=[softened → no-op, just instruction] │
│                                                                        │
│    "500g Carne Molida 8% grasa"  ← QUALIFIER + TRANSLATION TRAP        │
│      qty=500  unit=g  noun=Beef Mince  qualifier="8% fat"             │
│      → name: "Beef Mince"  comment: "Beef Mince: 8% fat"             │
│      ❌ WRONG: name: "Carne Molida 8% grasa"  (raw foreign + qualifier) │
│                                                                        │
│    "fine chopped onion"  ← THE CANONICAL TRAP                         │
│      noun=Brown Onion (id=onion)                                      │
│      pre-processes=[chop]  modifiers=[fine → cut=Brunoise]            │
│      ❌ WRONG: ingredient "Fine Chopped Onion".  That string is        │
│      half-verb half-modifier and pollutes both libraries.              │
│                                                                        │
│  PROCESS PLACEMENT RULE                                                │
│  ────────────────────────                                              │
│  Pre-processes from ingredient lines belong in a step where that       │
│  ingredient *first appears*.  You have two valid placements:           │
│    (a)  add a small dedicated prep step BEFORE the ingredient is       │
│         consumed (e.g. "Chop the onion").  Put the ingredient there,   │
│         the chop process there, and a Knife with cut=Brunoise as       │
│         the tool.  Use connectsTo to send it forward only if it        │
│         physically moves to another lane.                              │
│    (b)  if the ingredient is consumed in the same step where the       │
│         method describes the cut (e.g. "chop the onion finely and add  │
│         to the pan"), you may merge: use the cooking tool, and add     │
│         BOTH "chop" and the heat verb to processes.  This is rarer.    │
│  Default to (a) — explicit prep steps make better algorithms.          │
╰────────────────────────────────────────────────────────────────────────╯

╭─ Stage 3 ── Decompose every INSTRUCTION SENTENCE ──────────────────────╮
│  Cookbook instructions look like:                                      │
│    "Heat olive oil in a large pan over medium-high heat.  Add the      │
│     onions and sauté for 5 minutes until soft."                        │
│                                                                        │
│  This sentence yields TWO steps in two adjacent rows of one lane:     │
│    Step 1: ingredients=[Olive Oil 2 tbsp], processes=[],              │
│            tool=cast-iron-pan with opts.heat="Medium-High"            │
│    Step 2: ingredients=[<onion already in scope, do not repeat>],     │
│            processes=["sauté"],                                        │
│            tool=cast-iron-pan with opts.heat="Medium-High",            │
│            processParams={ _time:5, _timeUnit:"min" }                 │
│                                                                        │
│  Tokenisation rules                                                    │
│  ──────────────────                                                    │
│  • Verbs → processes.  Use the canonical process key (see PART C).    │
│    Multi-process steps get an array, e.g. ["sauté","reduce"].         │
│  • Time phrases  ("for 5 minutes", "8–10 mins", "until golden")       │
│    → _time + _timeUnit.  For ranges use the UPPER bound.  For         │
│    "until X" with no number, leave _time empty and put the criterion  │
│    in the instruction string.                                         │
│  • Temperature phrases  ("180 °C", "350 °F", "to a simmer")          │
│    → _temp.  ALWAYS convert °F to °C: C = round((F - 32) × 5/9).      │
│  • Heat phrases   ("medium-high heat", "low and slow")                │
│    → tool.opts.heat (only on stovetop tools).                         │
│  • Cut/size phrases  ("finely chop", "cut into matchsticks")          │
│    → tool.opts.cut (only on knife).                                   │
│  • Speed phrases  ("on speed 6", "high speed", "stir setting")        │
│    → tool.opts._speed (KitchenAid / Thermomix / Blender).             │
│  • Attachment phrases  ("with the dough hook", "whisk attachment")    │
│    → tool.opts.attachment (KitchenAid).                               │
│                                                                        │
│  "Meanwhile" / "While X cooks" / "At the same time" / "Set aside"      │
│  These are PARALLELISM SIGNALS.  Open or continue another lane.        │
│                                                                        │
│  "Drain the pasta into the pan" / "Pour the syrup over the whites" /   │
│  "Fold the meringue into the batter"                                   │
│  These are MERGE SIGNALS.  The source step keeps its tool;  add a      │
│  connectsTo entry pointing to the destination step ID, with side       │
│  "left" or "right" depending on relative lane position.                │
╰────────────────────────────────────────────────────────────────────────╯

╭─ Stage 4 ── Lay out the GRID ───────────────────────────────────────────╮
│  The grid is grid[row][lane].  Every row must have the SAME number of  │
│  lanes (pad with null).                                                │
│                                                                        │
│  Lane decisions                                                        │
│  ─────────────                                                         │
│  • 1 lane is fine for simple recipes (one cook stream, no parallels).  │
│  • Open a new lane when:                                               │
│       — there is a "Meanwhile…" or "While X is doing Y…" cue;          │
│       — there is a separate ingredient-group heading whose products    │
│         are produced independently (sauce, dough, marinade);           │
│       — a sub-preparation runs in a different appliance (oven warming  │
│         while you stir on the stove).                                  │
│  • laneNames[] should be human-readable labels, NOT generic ("Lane 1") │
│       — use "Boil Water", "Sauce", "Aromatics", "Dough", etc.          │
│  • Place each step in the lane where its tool / output naturally lives │
│       — pasta in the boil lane, sauce in the sauce lane, etc.          │
│  • Time order is rows.  Earlier rows happen first.  A lane can be      │
│    empty (null) for a row when nothing is happening there.             │
│                                                                        │
│  Ingredient placement                                                  │
│  ────────────────────                                                  │
│  Ingredients appear at the step where they are FIRST used (added,      │
│  measured, or transformed).  Do NOT repeat them in later steps unless  │
│  more is added later (e.g. "add half now, half later").                │
│                                                                        │
│  Module preparations                                                   │
│  ───────────────────                                                   │
│  If the source recipe references a sub-recipe ("see page 42 for the    │
│  pebre"), produce ONE preparation flagged isModule=true with a         │
│  laneNames of just that sub-name.  Otherwise isModule=false.           │
╰────────────────────────────────────────────────────────────────────────╯

╭─ Stage 5 ── Wire up CONNECTORS ─────────────────────────────────────────╮
│  Use connectsTo only when a step's output PHYSICALLY MOVES to another  │
│  lane.  Examples:                                                      │
│    • "Drain the pasta and add to the pan with the guanciale"          │
│         drain step  →  connectsTo target=add-pasta step  side=right   │
│    • "Pour the hot syrup over the whipped whites"                     │
│         syrup step  →  connectsTo target=combine step    side=left    │
│    • "Fold meringue gently into batter"                               │
│         meringue step → connectsTo target=fold step      side=left    │
│                                                                        │
│  side="left" if the destination lane is LEFT of the source lane;       │
│  side="right" if the destination lane is RIGHT of the source lane.     │
│  Use the targetId of the destination step.                             │
│                                                                        │
│  Do NOT connect within the same lane — vertical flow is implicit.      │
│  Do NOT use connectsTo for "set aside for later" — that is just time.  │
╰────────────────────────────────────────────────────────────────────────╯

═══════════════════════════════════════════════════════════════════════════
PART B — TOOL.OPTS  vs  processParams   (this is critical and easy to get wrong)
═══════════════════════════════════════════════════════════════════════════

The schema has TWO places where time and temperature can live:
   tool.opts          — fields built into the tool itself
   step.processParams — per-step kinetic data

Discriminator (memorise):
  IF the tool is one of:
       oven, thermomix, kitchenaid, sous-vide, blow-torch, fridge,
       freezer, sandwich-press, manual-knead
  THEN time / temp / speed go INTO tool.opts.
  AND  step.processParams should be empty or omitted.

  IF the tool is one of:
       reg-pot, stock-pot, saucepan, cast-iron-pot,
       reg-pan, cast-iron-pan, non-stick-pan, wok,
       knife, grater, sieve, colander, whisk, spatula,
       bowl, jug, scale, masher, squeezer,
       baking-tray, loaf-pan, cake-pan, quiche-pan, rolling-pin,
       oven-tray, oven-rack
  OR there is no tool (tool=null)
  THEN time / temp go into step.processParams.
  AND  tool.opts only contains its native options (heat, cut, type, grade).

Concrete shapes
───────────────
  oven.opts:        { _temp:180, _time:30, _timeUnit:"min", ovenMode:"Conventional" }
  thermomix.opts:   { _temp:80,  _time:10, _timeUnit:"min", _speed:"3" }
  kitchenaid.opts:  { _speed:"6", attachment:"Wire Whip", _time:5, _timeUnit:"min" }
  blender.opts:     { _speed:"Medium" }
  sous-vide.opts:   { _temp:60,  _time:2,  _timeUnit:"hr" }
  blow-torch.opts:  { heat:"High", _time:10, _timeUnit:"sec" }
  fridge.opts:      { _time:6, _timeUnit:"hr" }
  freezer.opts:     { _time:1, _timeUnit:"hr" }

  cast-iron-pan.opts:  { heat:"Medium-High" }       /* time goes in processParams */
  reg-pot.opts:        { heat:"High" }
  knife.opts:          { cut:"Brunoise" }
  grater.opts:         { grade:"Fine" }
  sieve.opts:          { type:"Fine Sieve" }
  bowl.opts:           {}
  whisk.opts:          {}

  step.processParams (when needed):
                       { _time:5, _timeUnit:"min" }
                       { _time:5, _timeUnit:"min", _temp:60 }   /* e.g. marinate at 25 °C */

═══════════════════════════════════════════════════════════════════════════
PART C — KNOWN MAPPINGS  (compressed reference)
═══════════════════════════════════════════════════════════════════════════

Verb → process key (use the RHS literally as a string in step.processes[]):
  Size/shape:  chop→chop, dice→dice, mince→mince, slice→slice, julienne→julienne,
               grate→grate, shred→shred, zest→zest, crush→crush, grind→grind
  Prep:        peel→peel, trim→trim, wash→wash, dry→dry, deseed→de-seed,
               shell→shell, bone→bone, weigh→weigh, measure→measure
  Hydration:   soak→soak, marinate→marinate, brine→brine, steep→steep, infuse→infuse
  Combine:     mix→mix, stir→stir, fold→fold, whisk→whisk, beat→beat, toss→toss,
               cream→cream, dissolve→dissolve
  Emulsion:    emulsify→emulsify, whip→whip, foam→foam
  Structure:   knead→knead, laminate→laminate, rest→rest, ferment→ferment, proof→proof
  Dry heat:    bake→bake, roast→roast, toast→toast, grill→grill, broil→broil,
               sear→sear (also "brown"), sauté→sauté, pan-roast→pan-roast
  Moist heat:  boil→boil, simmer→simmer, poach→poach, steam→steam, blanch→blanch
  Fat heat:    shallow-fry→shallow-fry, pan-fry→pan-fry (also "fry"), deep-fry→deep-fry, confit→confit
  Combination: braise→braise, stew→stew, pressure cook→pressure cook, pot-roast→pot-roast
  Reduce:      reduce→reduce (also "caramelise"), evaporate→evaporate,
               concentrate→concentrate, deglaze→deglaze, glaze→glaze
  Cool:        cool→cool, chill→chill, shock→shock, refrigerate→refrigerate, freeze→freeze
  Preserve:    pickle→pickle, smoke→smoke
  Finish:      season→season, garnish→garnish, dress→dress
  Assemble:    layer→layer, fill→fill, plate→plate, portion→portion, carve→carve
  Control:     taste→taste, adjust→adjust, filter→filter, strain→strain (also "drain"), sift→sift

Modifier → knife.opts.cut:
  fine, finely, very fine, minced            →  Brunoise
  fine brunoise, very very fine              →  Fine Brunoise
  small dice, small                          →  Small Dice
  medium, diced, cubed                       →  Medium Dice
  large, rough, roughly, coarse, chunks      →  Large Dice
  batonnet, batons, sticks, thick strips     →  Batonnet
  julienne, matchsticks, thin strips         →  Julienne
  fine julienne                              →  Fine Julienne
  chiffonade, ribbons                        →  Chiffonade
  thin, thinly, paper-thin                   →  Thin Cut
  thick, thickly                             →  Thick Cut
  (no qualifier, generic "slice")            →  Slice

Heat phrase → pan.opts.heat:
  very low, low and slow, gentle             →  Low
  medium-low, med-low                        →  Medium-Low
  moderate                                   →  Medium
  medium-high, med-high                      →  Medium-High
  high heat, high                            →  High
  off / no heat                              →  Off

Verb → default tool (when none mentioned explicitly):
  bake, roast, broil, grill, toast            →  oven
  sear, sauté, pan-fry, shallow-fry           →  cast-iron-pan
  stir-fry                                    →  wok
  deep-fry                                    →  stock-pot
  boil, blanch, parboil                       →  reg-pot
  simmer, poach, reduce                       →  saucepan
  steam                                       →  stock-pot
  braise, stew, confit                        →  cast-iron-pot
  chop, dice, mince, slice, julienne, etc.    →  knife
  whisk (manual)                              →  whisk
  beat, whip, knead, cream butter             →  kitchenaid
  fold, stir (off-heat)                       →  spatula
  mix (off-heat, no machine)                  →  bowl
  chill, refrigerate                          →  fridge
  freeze                                      →  freezer
  strain, sift, filter                        →  sieve
  drain                                       →  colander
  weigh, measure                              →  scale

Common ingredient synonyms (apply silently — the canonical NAME is on the right):
${serialiseSynonyms()}

Unit conversions Claude must apply silently:
  1 stick of butter   → 113 g
  1 knob of butter    → 15 g
  1 tablespoon (UK/AU)→ tbsp (15 ml)
  1 teaspoon          → tsp (5 ml)
  1 cup (US)          → cup (240 ml; 250 ml in AU — keep "cup" as unit)
  ¼, ½, ¾             → 0.25, 0.5, 0.75
  Temperatures in °F  → convert to °C  (round to nearest 5)
  Temperatures in gas mark → convert to °C (gas 4 ≈ 180, gas 6 ≈ 200, gas 7 ≈ 220)

═══════════════════════════════════════════════════════════════════════════
PART D — APP LIBRARIES (canonical IDs)
═══════════════════════════════════════════════════════════════════════════

INGREDIENTS  (use these exact ids in step.ingredients[*].ingredientId;  the
\`name\` field MUST be the canonical English name from the right of the arrow):
${serialiseIngredients(ctx.ingredientLibrary, ctx.customIngredients)}

TOOLS  (use these ids in step.tool.toolId; the \`name\` is the canonical name):
${serialiseTools(ctx.toolLibrary, ctx.customTools)}

PROCESSES  (use these strings literally in step.processes[]):
${serialiseProcesses(ctx.processLibrary, ctx.customProcesses)}

═══════════════════════════════════════════════════════════════════════════
PART E — UNKNOWN ITEMS — CANDIDATES, NEVER INVENTED IDS
═══════════════════════════════════════════════════════════════════════════

If a recipe uses an ingredient/tool/process you cannot match to the canonical
libraries above:

  ❌ DO NOT invent canonical-style ids like "merken" or "lemongrass".
  ❌ DO NOT pick the closest English word and pretend it matches.
  ✅ DO add an entry to the appropriate candidate list:
        customIngredientCandidates  /  customToolCandidates  /  customProcessCandidates
  ✅ DO put a placeholder ingredientId of the form "ci-PENDING-<slug>" or
     "ct-PENDING-<slug>" — the server will replace these with proper ids.
     Example: ingredientId="ci-PENDING-merken", name="Merkén".
  ✅ DO continue to use the placeholder consistently across all step
     references so the user can adopt-the-suggestion in one click.

Candidate fields:
  customIngredientCandidates: { name, category (best guess from libraries),
                                 unit (best guess: g/ml/ea/tsp/...), reason,
                                 confidence (0-1) }
  customToolCandidates:        { name, category (Stovetop/Oven/Appliances/...),
                                 reason, confidence }
  customProcessCandidates:     { name, category (e.g. "8. Dry Heat"),
                                 reason, confidence }

Be conservative — only suggest a candidate if NO existing item is close enough.

═══════════════════════════════════════════════════════════════════════════
PART F — IDS, FIELDS, AND OUTPUT INVARIANTS
═══════════════════════════════════════════════════════════════════════════

• Every step needs an \`id\` (8-char alphanumeric).  Make all IDs unique within
  the response.  The server will rewrite them, but they MUST be self-consistent
  for connectsTo to resolve.
• Every ingredient row needs a \`uid\` (also 8-char alphanumeric).
• Tools need a \`uid\` too.
• status MUST be "draft".
• flavour: estimate 0-100 on each axis.  Be conservative — the front-end
  will recompute live.  Heuristic: count obvious sweet/rich/spicy/acidic/umami
  ingredients, weight by quantity intuition, output integers.
• category: pick from { Entrees, Mains, Desserts, "Drinks & Cocktails" }.
• tags: 2-5 short lowercase tags (cuisine, technique, occasion).
• Eggs (id=egg / egg-white / egg-yolk):  always set isEgg=1, eggSize="XL"
  (the app's default), eggPart="whole" / "white" / "yolk".
• Multiplier ingredients (formula recipes like pavlova: "sugar = 1.8 × egg
  whites by weight"): set unit="×", multiplierRef=<other ingredient's uid>.
  Use this only when the source explicitly states a ratio formula.

═══════════════════════════════════════════════════════════════════════════
PART G — WORKED EXAMPLES
═══════════════════════════════════════════════════════════════════════════

Example 1 — the "fine chopped onion" trap

  SOURCE (cookbook):
    Ingredients
      • 200 g guanciale, finely chopped
      • 2 medium yellow onions, finely chopped
      • 4 cloves garlic, minced
      • 400 g spaghetti
    Method
      Bring a pot of salted water to a boil.  Meanwhile, sauté the
      guanciale in a cold cast-iron pan over medium-low heat until the
      fat renders.  Add the onions and garlic and cook for 5 minutes.
      Cook the spaghetti for 8 minutes, reserving 1 cup of pasta water.
      Drain and toss with the rendered guanciale.

  CORRECT DECOMPOSITION (key ideas, abbreviated):
    laneNames: ["Boil Water", "Sauté & Sauce"]
    Lane "Boil Water":
      Row 0: Step "Boil salted water"
             ingredients=[Water 2 l, Salt 10 g]
             tool=reg-pot opts={heat:"High"}
             processes=[]                    // boil is implicit; or add "boil"
      Row 2: Step "Cook spaghetti"
             ingredients=[Spaghetti (dry) 400 g]
             tool=reg-pot opts={heat:"High"}
             processes=["boil"]
             processParams={ _time:8, _timeUnit:"min" }
      Row 3: Step "Drain pasta, reserve water"
             ingredients=[Pasta Water 1 cup]
             tool=sieve opts={type:"Mesh Strainer"}
             processes=["strain"]
             connectsTo: → step "Toss" (right)
    Lane "Sauté & Sauce":
      Row 0: Step "Chop guanciale"
             ingredients=[Guanciale 200 g]
             tool=knife opts={cut:"Brunoise"}      ← from "finely chopped"
             processes=["chop"]                    ← the verb
      Row 0: (in same row, can be a single step OR split — split is cleaner)
             Step "Chop onions"
             ingredients=[Brown Onion 2 ea]       ← canonical ID, NOT "fine chopped onion"
             tool=knife opts={cut:"Brunoise"}
             processes=["chop"]
             (similarly garlic with cut="Brunoise" because "minced")
      Row 1: Step "Render guanciale"
             ingredients=[]  (already in scope)
             tool=cast-iron-pan opts={heat:"Medium-Low"}
             processes=["sauté"]
             processParams={ _time:4, _timeUnit:"min" }
      Row 2: Step "Add aromatics, cook"
             ingredients=[]
             tool=cast-iron-pan opts={heat:"Medium-Low"}
             processes=["sauté"]
             processParams={ _time:5, _timeUnit:"min" }
      Row 3: Step "Toss" (TARGET of drain connectsTo)
             ingredients=[]
             tool=cast-iron-pan opts={heat:"Off"}
             processes=["toss"]

  Key invariants demonstrated:
    • "Onion" not "Fine Chopped Onion".
    • "fine"/"finely" → cut="Brunoise" on the knife.
    • "Sauté for 5 minutes" → process=sauté, processParams={_time:5,_timeUnit:"min"}.
      (NOT in tool.opts — cast-iron-pan only stores heat.)
    • "Boil water" tool=reg-pot, opts has heat only;  cooking time on the
      pasta step is in processParams, not tool.opts.
    • "Meanwhile" opens a second lane.
    • "Drain → toss" is a connectsTo (physical migration of pasta).

Example 2 — bake a cake (the oven case)

  SOURCE:  "Bake at 350 °F for 30 minutes."

  Step:
    instruction: "Bake at 175 °C for 30 minutes."
    ingredients: []
    processes: ["bake"]
    tool: { toolId:"oven", name:"Oven",
            opts: { _temp:175, _time:30, _timeUnit:"min", ovenMode:"Conventional" } }
    processParams: {}                               ← omitted because oven holds time

Example 3 — light searing for 30 seconds

  SOURCE:  "Lightly sear the scallops on each side for 30 seconds."

  Step:
    instruction: "Lightly sear the scallops on each side for 30 seconds."
    ingredients: [Scallops 200 g]   (or empty if scallops in scope from ingredient list)
    processes: ["sear"]
    tool: { toolId:"cast-iron-pan", opts:{ heat:"High" } }
                                  ← "lightly" is a duration cue, not a heat cue;
                                    sear implies high heat;  short time is the "lightness"
    processParams: { _time:30, _timeUnit:"sec" }

Example 4 — simmer for 5 minutes

  SOURCE:  "Simmer the sauce for 5 minutes until thickened."

  Step:
    instruction: "Simmer the sauce for 5 minutes until thickened."
    processes: ["simmer","reduce"]    ← "until thickened" implies reduction
    tool: { toolId:"saucepan", opts:{ heat:"Medium-Low" } }
    processParams: { _time:5, _timeUnit:"min" }

Example 5 — Thermomix step (time goes IN tool.opts)

  SOURCE:  "Cook in the Thermomix at 100 °C, speed 3, for 5 minutes."

  Step:
    processes: ["infuse"]   /* or whatever fits the verb */
    tool: { toolId:"thermomix",
            opts: { _temp:100, _speed:"3", _time:5, _timeUnit:"min" } }
    processParams: {}

Example 6 — KitchenAid whip

  SOURCE:  "Whip the whites with the whisk attachment on speed 8 for 4 minutes
            until stiff peaks form."

  Step:
    processes: ["whip"]
    tool: { toolId:"kitchenaid",
            opts: { _speed:"8", attachment:"Wire Whip", _time:4, _timeUnit:"min" } }
    processParams: {}      ← all kinetics live in opts for kitchenaid

Example 7 — multiplier ingredient

  SOURCE:  "Use 1.8 × the weight of the egg whites in caster sugar."

  Step ingredients:
    [
      { uid:"abc12345", ingredientId:"egg-white", name:"Egg White", qty:5,
        unit:"ea", isEgg:1, eggSize:"XL", eggPart:"white" },
      { uid:"def67890", ingredientId:"caster-sugar", name:"Caster Sugar",
        qty:1.8, unit:"×", multiplierRef:"abc12345" }
    ]

═══════════════════════════════════════════════════════════════════════════
PART H — IMPORT MODES
═══════════════════════════════════════════════════════════════════════════

The user can request three modes:
  • "preserve"     — closely follow the source structure.  One lane unless
                     the source clearly signals parallelism.  Conservative.
  • "smart"        (default) — apply your full decomposition judgement.
                     Open lanes for "meanwhile", split prep from cook,
                     promote ingredient groups to lanes when productive.
  • "professional" — restructure aggressively for execution efficiency.
                     Pull out reusable sub-preparations (sauces, doughs)
                     as isModule=true.  Maximise parallel lanes.

═══════════════════════════════════════════════════════════════════════════
PART I — REASONING WORKFLOW (MANDATORY ORDER)
═══════════════════════════════════════════════════════════════════════════

In the thinking block, before the tool call, walk through:
  1. Language detected, source sections identified.
  2. Ingredient list → for each line, [qty, unit, canonical id, pre-processes,
     modifiers, tool-opt hints].  If unknown → candidate row (ci-PENDING-…).
  3. Instruction list → for each sentence:
        verbs found (→ processes),
        time/temp phrases (→ tool.opts vs processParams via the discriminator),
        heat/cut/speed phrases (→ tool.opts),
        parallelism cues ("meanwhile") → lane decisions,
        merge cues ("drain into", "fold into") → connectsTo plans.
  4. Lane plan: list lane names left-to-right, what flows in each.
  5. Row plan: which step happens in which row (time order).  Mark the
     connectsTo edges.
  6. Flavour estimate: list dominant axis-driving ingredients, score 0-100.

THEN call submit_recipe_draft exactly once.  Do not write anything outside
the tool call.

If the source is too ambiguous to parse (e.g. just a title, or unintelligible),
call submit_recipe_draft with ok=false and a clear message in the message
field — but try hard first.  Even a partial draft is more useful than nothing.
`;


/*  ═══════════════════════════════════════════════════════════════════════════
    SECTION 6 — TOOL-USE SCHEMA  (the structured-output contract)
    Anthropic's tool use forces JSON that matches this schema.  We declare
    the fields exactly the way the front-end expects them.  Where a value
    must come from a fixed enum we declare it.
    ═════════════════════════════════════════════════════════════════════════ */

const submitRecipeDraftTool = {
  name: "submit_recipe_draft",
  description: "Submit the structured recipe draft.  Call this exactly once.",
  input_schema: {
    type: "object",
    additionalProperties: false,
    required: ["ok"],
    properties: {
      ok: { type: "boolean" },
      message: { type: "string", description: "Human-readable note when ok=false, or a short summary when ok=true." },
      confidence: { type: "number", minimum: 0, maximum: 1 },
      recipeDraft: {
        type: "object",
        additionalProperties: false,
        required: ["name", "servings", "preparations"],
        properties: {
          name:        { type: "string", minLength: 1 },
          description: { type: "string" },
          servings:    { type: "integer", minimum: 1, maximum: 100 },
          tags:        { type: "array", items: { type: "string" }, maxItems: 8 },
          category:    { type: "string", enum: RECIPE_CATS },
          flavour: {
            type: "object",
            additionalProperties: false,
            required: FLAVOUR_AXES,
            properties: Object.fromEntries(
              FLAVOUR_AXES.map(a => [a, { type: "integer", minimum: 0, maximum: 100 }])
            ),
          },
          preparations: {
            type: "array",
            minItems: 1,
            items: {
              type: "object",
              additionalProperties: false,
              required: ["name", "isModule", "laneNames", "grid"],
              properties: {
                id:          { type: "string" },
                name:        { type: "string", minLength: 1 },
                comment:     { type: "string" },
                isModule:    { type: "boolean" },
                laneNames:   { type: "array", items: { type: "string" }, minItems: 1 },
                grid: {
                  /* grid[row][lane] : null | Step */
                  type: "array",
                  description: "2D array [row][lane].  Cells are either null or a Step object.",
                  items: {
                    type: "array",
                    items: {
                      anyOf: [
                        { type: "null" },
                        {
                          type: "object",
                          additionalProperties: false,
                          required: ["id", "title", "ingredients", "processes"],
                          properties: {
                            id:          { type: "string", minLength: 1 },
                            title:       { type: "string" },
                            instruction: { type: "string" },
                            ingredients: {
                              type: "array",
                              items: {
                                type: "object",
                                additionalProperties: false,
                                required: ["uid", "ingredientId", "name", "qty", "unit"],
                                properties: {
                                  uid:           { type: "string" },
                                  ingredientId:  { type: "string" },
                                  name:          { type: "string" },
                                  qty:           { type: "number" },
                                  unit:          { type: "string", enum: ALLOWED_UNITS },
                                  isEgg:         { type: "integer", enum: [0, 1] },
                                  eggSize:       { type: "string", enum: ["Small","Medium","Large","XL","Jumbo"] },
                                  eggPart:       { type: "string", enum: ["whole","white","yolk"] },
                                  isChilli:      { type: "integer", enum: [0, 1] },
                                  chilliHeat:    { type: "string", enum: ["m","md","h","vh","ex"] },
                                  multiplierRef: { type: "string" },
                                },
                              },
                            },
                            processes: {
                              type: "array",
                              items: { type: "string" },
                            },
                            tool: {
                              anyOf: [
                                { type: "null" },
                                {
                                  type: "object",
                                  additionalProperties: false,
                                  required: ["uid", "toolId", "name", "opts"],
                                  properties: {
                                    uid:    { type: "string" },
                                    toolId: { type: "string" },
                                    name:   { type: "string" },
                                    opts:   {
                                      type: "object",
                                      additionalProperties: true,
                                      /* permissive — we validate enum values post-hoc */
                                    },
                                  },
                                },
                              ],
                            },
                            processParams: {
                              type: "object",
                              additionalProperties: false,
                              properties: {
                                _time:     { type: "number", minimum: 0 },
                                _timeUnit: { type: "string", enum: TIME_UNITS },
                                _temp:     { type: "number" },
                              },
                            },
                            connectsTo: {
                              type: "array",
                              items: {
                                type: "object",
                                additionalProperties: false,
                                required: ["targetId", "side"],
                                properties: {
                                  targetId: { type: "string" },
                                  side:     { type: "string", enum: ["left", "right"] },
                                },
                              },
                            },
                          },
                        },
                      ],
                    },
                  },
                },
                anchorPrepId: { type: "string" },
                anchorStepId: { type: "string" },
              },
            },
          },
        },
      },
      customIngredientCandidates: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          required: ["name"],
          properties: {
            name:       { type: "string" },
            category:   { type: "string" },
            unit:       { type: "string" },
            reason:     { type: "string" },
            confidence: { type: "number", minimum: 0, maximum: 1 },
          },
        },
      },
      customToolCandidates: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          required: ["name"],
          properties: {
            name:       { type: "string" },
            category:   { type: "string" },
            reason:     { type: "string" },
            confidence: { type: "number", minimum: 0, maximum: 1 },
          },
        },
      },
      customProcessCandidates: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          required: ["name"],
          properties: {
            name:       { type: "string" },
            category:   { type: "string" },
            reason:     { type: "string" },
            confidence: { type: "number", minimum: 0, maximum: 1 },
          },
        },
      },
      warnings:    { type: "array", items: { type: "string" } },
      assumptions: { type: "array", items: { type: "string" } },
      unmapped:    { type: "array", items: { type: "string" } },
    },
  },
};

/*  ═══════════════════════════════════════════════════════════════════════════
    SECTION 7 — REQUEST VALIDATION  (zod)
    ═════════════════════════════════════════════════════════════════════════ */

const RequestSchema = z.object({
  sourceText: z.string().min(10).max(60000),
  hints: z.object({
    title:      z.string().optional(),
    servings:   z.number().int().min(1).max(100).optional(),
    language:   z.enum(["auto","en","es","fr"]).default("auto"),
    importMode: z.enum(["smart","preserve","professional"]).default("smart"),
  }),
  appContext: z.object({
    ingredientLibrary: z.array(z.object({
      id:  z.string(),
      n:   z.string(),
      cat: z.string().optional(),
    })).optional(),
    toolLibrary: z.array(z.object({
      id: z.string(),
      n:  z.string(),
    })).optional(),
    processLibrary:    z.array(z.string()).optional(),
    customIngredients: z.any().optional(),
    customTools:       z.any().optional(),
    customProcesses:   z.any().optional(),
    schemaVersion:     z.number().default(1),
  }),
});

/*  ═══════════════════════════════════════════════════════════════════════════
    SECTION 8 — PRE-PROCESSING  (light source-text normalisation)
    Keep this conservative.  Heavy lifting is in the model.
    ═════════════════════════════════════════════════════════════════════════ */

const normaliseSourceText = (txt) => {
  return txt
    /* unicode dashes → ascii hyphen */
    .replace(/[\u2013\u2014]/g, "-")
    /* curly quotes → straight */
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, '"')
    /* nbsp → space */
    .replace(/\u00A0/g, " ")
    /* unicode fractions */
    .replace(/¼/g, "1/4").replace(/½/g, "1/2").replace(/¾/g, "3/4")
    .replace(/⅓/g, "1/3").replace(/⅔/g, "2/3")
    .replace(/⅕/g, "1/5").replace(/⅖/g, "2/5").replace(/⅗/g, "3/5").replace(/⅘/g, "4/5")
    .replace(/⅙/g, "1/6").replace(/⅚/g, "5/6")
    .replace(/⅛/g, "1/8").replace(/⅜/g, "3/8").replace(/⅝/g, "5/8").replace(/⅞/g, "7/8")
    /* multi-newline → double newline (preserve paragraph structure) */
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    /* trim each line */
    .split("\n").map(l => l.trim()).join("\n")
    .trim();
};

const buildUserMessage = (sourceText, hints) => {
  const parts = [];
  if (hints.title)    parts.push(`RECIPE NAME (hint from user): ${hints.title}`);
  if (hints.servings) parts.push(`SERVINGS (hint from user): ${hints.servings}`);
  if (hints.language && hints.language !== "auto") parts.push(`LANGUAGE: ${hints.language}`);
  parts.push(`IMPORT MODE: ${hints.importMode}`);
  parts.push("");
  parts.push("══ RECIPE SOURCE ══");
  parts.push(normaliseSourceText(sourceText));
  parts.push("══ END SOURCE ══");
  parts.push("");
  parts.push("Walk the 5-stage pipeline in the thinking block, then call submit_recipe_draft exactly once.");
  return parts.join("\n");
};


/*  ═══════════════════════════════════════════════════════════════════════════
    SECTION 9 — POST-PROCESSING
    The LLM's output is treated as advisory.  We:
      • re-issue all UIDs server-side (deterministic 8-char alnum)
      • remap connectsTo references through the new id table
      • rectangularise the grid (every row gets the same lane count)
      • reconcile knife.cut, pan.heat, oven.ovenMode against allowed enums
      • move misplaced time/temp from tool.opts to processParams (or vice versa)
        depending on the discriminator
      • rewrite ci-PENDING-* placeholders into ci-<random> ids and emit a
        candidate row if the LLM forgot
      • estimate flavour from ingredients if the LLM punted (the 50/50/0/20/20
        default we keep seeing)
    ═════════════════════════════════════════════════════════════════════════ */

const newUid = () => Math.random().toString(36).slice(2, 10);

const TOOLS_WITH_INTERNAL_KINETICS = new Set([
  "oven", "thermomix", "kitchenaid", "sous-vide", "blow-torch",
  "fridge", "freezer", "sandwich-press", "manual-knead", "blender",
]);

const STOVETOP_TOOLS = new Set([
  "reg-pot", "stock-pot", "saucepan", "cast-iron-pot",
  "reg-pan", "cast-iron-pan", "non-stick-pan", "wok", "stove",
]);

/* Reissue all UIDs and remap connectsTo references.
   stepIdMap and ingUidMap are kept separate so a step id and an ingredient
   uid can never collide (they live in different namespaces in the schema). */
const reissueIds = (recipeDraft) => {
  if (!recipeDraft || !Array.isArray(recipeDraft.preparations)) return recipeDraft;
  const stepIdMap = {};   /* old step id → new step id */
  const ingUidMap = {};   /* old ingredient uid → new ingredient uid */
  const newStepIds = new Set();

  recipeDraft.id = newUid();
  recipeDraft.status = "draft";

  for (const prep of recipeDraft.preparations) {
    prep.id = newUid();
    if (!Array.isArray(prep.grid)) { prep.grid = []; continue; }

    /* Pass 1: assign new step ids and ingredient/tool uids. */
    for (const row of prep.grid) {
      if (!Array.isArray(row)) continue;
      for (const cell of row) {
        if (!cell) continue;
        const oldId = cell.id;
        cell.id = newUid();
        newStepIds.add(cell.id);
        if (oldId) stepIdMap[oldId] = cell.id;

        if (Array.isArray(cell.ingredients)) {
          for (const ing of cell.ingredients) {
            const oldUid = ing.uid;
            ing.uid = newUid();
            if (oldUid) ingUidMap[oldUid] = ing.uid;
          }
        }
        if (cell.tool) cell.tool.uid = newUid();
      }
    }

    /* Pass 2: remap connectsTo and multiplierRef references. */
    for (const row of prep.grid) {
      if (!Array.isArray(row)) continue;
      for (const cell of row) {
        if (!cell) continue;
        if (Array.isArray(cell.connectsTo)) {
          cell.connectsTo = cell.connectsTo
            .map(c => ({ ...c, targetId: stepIdMap[c.targetId] || c.targetId }))
            /* Drop dangling references (target not a known step id). */
            .filter(c => newStepIds.has(c.targetId));
        }
        if (Array.isArray(cell.ingredients)) {
          for (const ing of cell.ingredients) {
            if (ing.multiplierRef && ingUidMap[ing.multiplierRef]) {
              ing.multiplierRef = ingUidMap[ing.multiplierRef];
            } else if (ing.multiplierRef) {
              /* Dangling — clear it so the FE doesn't break. */
              delete ing.multiplierRef;
              if (ing.unit === "×") ing.unit = "g";  /* fallback */
            }
          }
        }
      }
    }

    /* Pass 3: rectangularise the grid. */
    const maxLanes = Math.max(prep.laneNames?.length || 1,
      ...prep.grid.map(r => Array.isArray(r) ? r.length : 0), 1);
    /* Pad laneNames if short */
    while ((prep.laneNames?.length || 0) < maxLanes) {
      (prep.laneNames ??= []).push(`Lane ${prep.laneNames.length + 1}`);
    }
    for (const row of prep.grid) {
      if (!Array.isArray(row)) continue;
      while (row.length < maxLanes) row.push(null);
      if (row.length > maxLanes) row.length = maxLanes;
    }
  }
  return recipeDraft;
};

/* Move time/temp into the right home for a step.
   Mutates step.tool.opts and step.processParams in place. */
const reconcileKinetics = (step) => {
  if (!step) return;
  const tool = step.tool;
  const tid = tool?.toolId || "";

  if (TOOLS_WITH_INTERNAL_KINETICS.has(tid)) {
    /* Time/temp belong in tool.opts.  Pull from processParams if mis-placed. */
    const pp = step.processParams || {};
    tool.opts ??= {};
    if (pp._time     != null && tool.opts._time     == null) tool.opts._time     = pp._time;
    if (pp._timeUnit != null && tool.opts._timeUnit == null) tool.opts._timeUnit = pp._timeUnit;
    if (pp._temp     != null && tool.opts._temp     == null) tool.opts._temp     = pp._temp;
    /* Strip processParams to keep the cell clean. */
    if (Object.keys(pp).length) {
      delete pp._time; delete pp._timeUnit; delete pp._temp;
      if (Object.keys(pp).length === 0) delete step.processParams;
    }
    /* For oven, ensure mode default. */
    if (tid === "oven" && !tool.opts.ovenMode) tool.opts.ovenMode = "Conventional";
    /* For kitchenaid, ensure attachment default. */
    if (tid === "kitchenaid" && !tool.opts.attachment) tool.opts.attachment = "Wire Whip";
  } else if (STOVETOP_TOOLS.has(tid) || !tool) {
    /* Time/temp belong in processParams.  Pull from tool.opts if mis-placed. */
    if (tool) {
      const opts = tool.opts || {};
      step.processParams ??= {};
      if (opts._time     != null && step.processParams._time     == null) step.processParams._time     = opts._time;
      if (opts._timeUnit != null && step.processParams._timeUnit == null) step.processParams._timeUnit = opts._timeUnit;
      if (opts._temp     != null && step.processParams._temp     == null) step.processParams._temp     = opts._temp;
      delete opts._time; delete opts._timeUnit; delete opts._temp;
      tool.opts = opts;
    }
    /* Cleanup empty processParams. */
    if (step.processParams && Object.keys(step.processParams).length === 0) {
      delete step.processParams;
    }
  }
};

/* Validate enum values on tool.opts and step ingredients.  Coerce common
   slips (case differences, "medium high" → "Medium-High"). */
const reconcileEnums = (step) => {
  if (!step?.tool?.opts) return;
  const t = step.tool;
  const o = t.opts;

  const tryMatch = (val, allowed) => {
    if (val == null) return val;
    const s = String(val).trim();
    /* exact */
    if (allowed.includes(s)) return s;
    /* case-insensitive */
    const ci = allowed.find(a => a.toLowerCase() === s.toLowerCase());
    if (ci) return ci;
    /* loose: hyphens vs spaces */
    const norm = s.replace(/\s+/g, "-").toLowerCase();
    const lo = allowed.find(a => a.replace(/\s+/g, "-").toLowerCase() === norm);
    return lo || null;
  };

  if (o.heat        != null) o.heat        = tryMatch(o.heat,        HEAT_OPTS)    ?? "Medium";
  if (o.cut         != null) o.cut         = tryMatch(o.cut,         KNIFE_CUTS)   ?? "Medium Dice";
  if (o.ovenMode    != null) o.ovenMode    = tryMatch(o.ovenMode,    OVEN_MODES)   ?? "Conventional";
  if (o.attachment  != null) o.attachment  = tryMatch(o.attachment,  KA_ATTACH)    ?? "Wire Whip";
  if (o.grade       != null) o.grade       = tryMatch(o.grade,       GRATER_GRADES)?? "Fine";
  if (o.type        != null) o.type        = tryMatch(o.type,        SIEVE_TYPES)  ?? "Mesh Strainer";

  if (o._timeUnit   != null) o._timeUnit   = tryMatch(o._timeUnit, TIME_UNITS)     ?? "min";

  /* Speed validation depends on the tool. */
  if (o._speed != null) {
    const allowed = t.toolId === "kitchenaid" ? KA_SPEEDS
                  : t.toolId === "thermomix"  ? TM_SPEEDS
                  : t.toolId === "blender"    ? BLENDER_SPEEDS
                  : null;
    if (allowed) o._speed = tryMatch(o._speed, allowed) ?? allowed[Math.floor(allowed.length / 2)];
  }

  if (step.processParams?._timeUnit != null) {
    step.processParams._timeUnit =
      tryMatch(step.processParams._timeUnit, TIME_UNITS) ?? "min";
  }
};

/* Rewrite "ci-PENDING-*" / "ct-PENDING-*" placeholders into proper random
   ids.  Emit a candidate row if the candidate list missed it. */
const reconcilePending = (out) => {
  const ingMap = {};   /* PENDING-slug → ci-<random> */
  const toolMap = {};

  const ingFromCandidate = (slug) => {
    const c = (out.customIngredientCandidates || []).find(x =>
      x.name?.toLowerCase().replace(/\s+/g, "-") === slug.toLowerCase());
    return c || null;
  };
  const toolFromCandidate = (slug) => {
    const c = (out.customToolCandidates || []).find(x =>
      x.name?.toLowerCase().replace(/\s+/g, "-") === slug.toLowerCase());
    return c || null;
  };

  if (!out.recipeDraft?.preparations) return out;

  for (const prep of out.recipeDraft.preparations) {
    for (const row of prep.grid || []) {
      if (!Array.isArray(row)) continue;
      for (const cell of row) {
        if (!cell) continue;
        for (const ing of cell.ingredients || []) {
          if (typeof ing.ingredientId === "string" && ing.ingredientId.startsWith("ci-PENDING-")) {
            const slug = ing.ingredientId.slice("ci-PENDING-".length);
            if (!ingMap[slug]) {
              ingMap[slug] = `ci-${newUid()}`;
              if (!ingFromCandidate(slug)) {
                /* Synthesise a candidate row from the ingredient's name. */
                (out.customIngredientCandidates ??= []).push({
                  name:       ing.name || slug,
                  category:   "Misc",
                  unit:       ing.unit || "g",
                  reason:     "Auto-promoted from PENDING placeholder.",
                  confidence: 0.5,
                });
              }
            }
            ing.ingredientId = ingMap[slug];
          }
        }
        if (cell.tool && typeof cell.tool.toolId === "string" &&
            cell.tool.toolId.startsWith("ct-PENDING-")) {
          const slug = cell.tool.toolId.slice("ct-PENDING-".length);
          if (!toolMap[slug]) {
            toolMap[slug] = `ct-${newUid()}`;
            if (!toolFromCandidate(slug)) {
              (out.customToolCandidates ??= []).push({
                name:       cell.tool.name || slug,
                category:   "Manual Tools",
                reason:     "Auto-promoted from PENDING placeholder.",
                confidence: 0.5,
              });
            }
          }
          cell.tool.toolId = toolMap[slug];
        }
      }
    }
  }
  return out;
};

/* Resolve an ingredient name to a FLAVOUR_HINTS entry.
   Tries (in order):
     1. Direct normalised-lowercase lookup  (English canonical names)
     2. INGREDIENT_SYNONYMS  (English regional variants → canonical)
     3. INGREDIENT_TRANSLATIONS  (es/fr/it → English canonical)
   Returns the hint object or undefined. */
const resolveFlavorHint = (rawName) => {
  const key = (rawName || "").toLowerCase().trim();
  if (FLAVOUR_HINTS[key]) return FLAVOUR_HINTS[key];
  const viaRegional = INGREDIENT_SYNONYMS[key];
  if (viaRegional && FLAVOUR_HINTS[viaRegional.toLowerCase()]) return FLAVOUR_HINTS[viaRegional.toLowerCase()];
  const viaTranslation = INGREDIENT_TRANSLATIONS[key];
  if (viaTranslation && FLAVOUR_HINTS[viaTranslation.toLowerCase()]) return FLAVOUR_HINTS[viaTranslation.toLowerCase()];
  return undefined;
};

/* Estimate flavour from ingredients (only used if LLM gave the boring
   default 50/50/0/20/20 — we treat that as "didn't try"). */
const estimateFlavour = (recipeDraft) => {
  const score = { sweetness:0, richness:0, spiciness:0, acidity:0, umami:0 };
  let total = 0;
  for (const prep of recipeDraft.preparations || []) {
    for (const row of prep.grid || []) {
      if (!Array.isArray(row)) continue;
      for (const cell of row) {
        if (!cell?.ingredients) continue;
        for (const ing of cell.ingredients) {
          const hint = resolveFlavorHint(ing.name);
          if (!hint) continue;
          /* weight by qty in grams-ish terms: 100 g equivalent = weight 1.
             countables: 1 ea ≈ 50 g, tsp ≈ 5 g, tbsp ≈ 15 g, cup ≈ 240 g. */
          let mass;
          switch (ing.unit) {
            case "g":   mass = ing.qty;          break;
            case "kg":  mass = ing.qty * 1000;   break;
            case "ml":  mass = ing.qty;          break;
            case "l":
            case "L":   mass = ing.qty * 1000;   break;
            case "tsp": mass = ing.qty * 5;      break;
            case "tbsp":mass = ing.qty * 15;     break;
            case "cup": mass = ing.qty * 240;    break;
            case "ea":  mass = ing.qty * 50;     break;
            default:    mass = ing.qty * 10;
          }
          /* tame outliers — saturate at 500 g equivalent */
          mass = Math.min(mass, 500);
          score.sweetness += (hint.s || 0) * mass;
          score.richness  += (hint.r || 0) * mass;
          score.spiciness += (hint.p || 0) * mass;
          score.acidity   += (hint.a || 0) * mass;
          score.umami     += (hint.u || 0) * mass;
          total += mass;
        }
      }
    }
  }
  if (total === 0) return null;
  /* normalise to 0-100, with a gentle compression so single dominant
     ingredients don't peg the axis */
  const norm = (v) => Math.round(Math.min(100, (v / total) * 200));
  return {
    sweetness: norm(score.sweetness),
    richness:  norm(score.richness),
    spiciness: norm(score.spiciness),
    acidity:   norm(score.acidity),
    umami:     norm(score.umami),
  };
};

/* The default flavour the v1 prompt produced verbatim — we treat exact match
   as "the LLM punted, please estimate from ingredients". */
const isFlavourPunt = (f) => f &&
  f.sweetness === 50 && f.richness === 50 && f.spiciness === 0 &&
  f.acidity === 20  && f.umami === 20;

/* Normalise a unit string to one of the ALLOWED_UNITS.  If the LLM emitted
   "tablespoon" or "teaspoons" or "ounces", map it sensibly. */
const UNIT_ALIASES = {
  "teaspoon":   "tsp",  "teaspoons":  "tsp",
  "tablespoon": "tbsp", "tablespoons":"tbsp",
  "cups":       "cup",
  "gram":       "g",    "grams":      "g",     "gr":     "g",
  "kilogram":   "kg",   "kilograms":  "kg",    "kilo":   "kg",
  "millilitre": "ml",   "millilitres":"ml",    "milliliter":"ml", "milliliters":"ml",
  "litre":      "l",    "litres":     "l",     "liter":  "l",     "liters":"l",
  "ounce":      "g",    "ounces":     "g",     "oz":     "g",     /* will warn — better than failing */
  "pound":      "kg",   "pounds":     "kg",    "lb":     "kg",
  "each":       "ea",   "whole":      "ea",
  "clove":      "cloves",
  "stalk":      "stalks",
  "pinches":    "pinch",
  "bunches":    "bunch",
  "sprigs":     "sprig",
  "slices":     "slice",
  "strips":     "strip",
};
const normaliseUnit = (u) => {
  if (!u) return "g";
  const s = String(u).trim();
  if (ALLOWED_UNITS.includes(s)) return s;
  const lo = s.toLowerCase();
  if (UNIT_ALIASES[lo]) return UNIT_ALIASES[lo];
  /* "L" (capital) was added to ALLOWED_UNITS; lowercase form too */
  if (lo === "l") return "l";
  /* Fall back gracefully */
  return "g";
};

/* Walk every step and reconcile kinetics + enums. */

/* ─────────────────────────────────────────────────────────────────────────
   SERVER-SIDE INGREDIENT NAME TRANSLATION
   Guaranteed fallback: even if the LLM ignores the Stage 1 mandate and
   emits a foreign-language name, this pass rewrites it to the canonical
   English library name by walking INGREDIENT_TRANSLATIONS then
   INGREDIENT_SYNONYMS.  Called from postProcess() on every ingredient.
   ──────────────────────────────────────────────────────────────────────── */
const QUALIFIER_PATTERNS = [
  /* Spanish */
  /\b\d+\s*%\s*(grasa|materia grasa|fat|mg)/i,
  /\b(extra\s+)?magra?\b/i,
  /\bbajo\s+en\s+grasa\b/i,
  /\bsin\s+hueso\b/i,
  /\bcon\s+hueso\b/i,
  /\bsin\s+piel\b/i,
  /\bcon\s+piel\b/i,
  /\bdeshuesad[ao]\b/i,
  /\bentera\b/i,
  /\benvasad[ao]\b/i,
  /\bfresc[ao]\b/i,
  /\bcongelad[ao]\b/i,
  /* French */
  /\bmaigre\b/i,
  /\bsans\s+os\b/i,
  /\bfraîche?\b/i,
  /* English */
  /\b\d+\s*%\s*(lean|fat|protein)\b/i,
  /\bextra\s+(lean|fine|thick)\b/i,
  /\bboneless\b/i,
  /\bskinless\b/i,
  /\bfresh\b/i,
  /\bfrozen\b/i,
  /\bwhole\s+milk\b/i,
  /\bfull.?fat\b/i,
  /\blow.?fat\b/i,
];

/**
 * Given an ingredient name (possibly in a foreign language or with quality
 * qualifiers), returns:
 *   { name: <canonical English name>, qualifier: <stripped qualifier or ""> }
 *
 * Resolution order:
 *   1. Strip trailing qualifiers first (e.g. "Carne Molida 8% grasa" → name="Carne Molida", qual="8% grasa")
 *   2. Lowercase + trim
 *   3. Check INGREDIENT_TRANSLATIONS (foreign → English canonical)
 *   4. Check INGREDIENT_SYNONYMS (English regional → canonical)
 *   5. If still unknown, return the stripped name as-is (possibly foreign)
 *
 * The qualifier (if any) is returned separately so the caller can append it
 * to the step's comment field.
 */
const resolveIngredientName = (rawName) => {
  if (!rawName) return { name: rawName, qualifier: "" };

  let name = rawName.trim();
  let qualifier = "";

  /* --- Pass 1: extract trailing qualifiers -------------------------------- */
  /* Strategy: for each qualifier pattern, test the WHOLE name.  If it matches
     anywhere, separate the qualifier fragment from the core noun.
     We work on a working copy so we can strip multiple qualifiers. */
  let working = name;
  const qualParts = [];

  for (const pat of QUALIFIER_PATTERNS) {
    const m = working.match(pat);
    if (m) {
      qualParts.push(m[0].trim());
      working = working.replace(pat, " ").replace(/\s{2,}/g, " ").trim();
      /* Strip leading/trailing punctuation left behind */
      working = working.replace(/^[,;:\-]+|[,;:\-]+$/g, "").trim();
    }
  }
  /* Also strip bare percentage patterns not caught above: "8%" alone */
  working = working.replace(/\b\d+\s*%\b/g, (m) => { qualParts.push(m); return ""; })
                   .replace(/\s{2,}/g, " ").trim()
                   .replace(/^[,;:\-]+|[,;:\-]+$/g, "").trim();

  if (qualParts.length > 0) {
    qualifier = qualParts.join(", ");
    name = working || name; /* fall back to original if stripping left nothing */
  }

  /* --- Pass 2: translate / resolve to canonical English name -------------- */
  const key = name.toLowerCase().trim();
  const translated =
    INGREDIENT_TRANSLATIONS[key] ||
    INGREDIENT_SYNONYMS[key]     ||
    null;

  if (translated) name = translated;

  return { name, qualifier };
};

/**
 * Walk every ingredient in every step of the draft and:
 *   a) resolve the name to canonical English (via resolveIngredientName)
 *   b) if a qualifier was stripped, append it to the step's comment field
 *
 * Mutates the recipeDraft in-place.
 */
const translateIngredientNames = (recipeDraft) => {
  for (const prep of recipeDraft.preparations || []) {
    for (const row of prep.grid || []) {
      if (!Array.isArray(row)) continue;
      for (const step of row) {
        if (!step?.ingredients) continue;
        const extraComments = [];
        for (const ing of step.ingredients) {
          const { name, qualifier } = resolveIngredientName(ing.name);
          ing.name = name;
          if (qualifier) {
            extraComments.push(`${name}: ${qualifier}`);
          }
        }
        if (extraComments.length > 0) {
          const existing = (step.comment || "").trim();
          step.comment = [existing, ...extraComments].filter(Boolean).join(" | ");
        }
      }
    }
  }
};

const postProcess = (out) => {
  if (!out?.recipeDraft) return out;

  /* Pre-id pass: pending placeholders before id rewrite */
  reconcilePending(out);

  /* Reissue ids and rectangularise. */
  reissueIds(out.recipeDraft);

  /* Per-step reconciliation. */
  for (const prep of out.recipeDraft.preparations || []) {
    for (const row of prep.grid || []) {
      if (!Array.isArray(row)) continue;
      for (const cell of row) {
        if (!cell) continue;
        reconcileKinetics(cell);
        reconcileEnums(cell);
        /* Defensive unit normalisation. */
        for (const ing of (cell.ingredients || [])) {
          ing.unit = normaliseUnit(ing.unit);
          if (typeof ing.qty !== "number" || !isFinite(ing.qty)) ing.qty = 1;
        }
        /* Sensible defaults for required fields */
        cell.ingredients ??= [];
        cell.processes   ??= [];
        cell.instruction ??= "";
        cell.title       ??= "Step";
      }
    }
  }

  /* Server-side ingredient name translation + qualifier stripping.
     Guaranteed fallback even when the LLM emits foreign-language names. */
  translateIngredientNames(out.recipeDraft);

  /* Flavour: estimate if punted or missing. */
  if (!out.recipeDraft.flavour || isFlavourPunt(out.recipeDraft.flavour)) {
    const est = estimateFlavour(out.recipeDraft);
    if (est) out.recipeDraft.flavour = est;
    else out.recipeDraft.flavour ??= { sweetness:50, richness:50, spiciness:0, acidity:20, umami:20 };
  }

  /* Normalise array fields on top-level. */
  out.warnings    ??= [];
  out.assumptions ??= [];
  out.unmapped    ??= [];
  out.customIngredientCandidates ??= [];
  out.customToolCandidates       ??= [];
  out.customProcessCandidates    ??= [];

  /* Make sure category is sensible. */
  if (out.recipeDraft.category && !RECIPE_CATS.includes(out.recipeDraft.category)) {
    out.recipeDraft.category = "Mains";
  }

  return out;
};


/*  ═══════════════════════════════════════════════════════════════════════════
    SECTION 10 — EXPRESS APP + MAIN ENDPOINT
    ═════════════════════════════════════════════════════════════════════════ */

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(cors({
  origin:  ALLOWED_ORIGINS[0] === "*" ? true : ALLOWED_ORIGINS,
  methods: ["POST", "OPTIONS", "GET"],
}));

/* Extract the tool_use block from Anthropic's response. */
const extractToolUse = (response) => {
  const block = (response.content || []).find(
    b => b.type === "tool_use" && b.name === "submit_recipe_draft"
  );
  return block?.input || null;
};

app.post("/api/import-pro/parse", async (req, res) => {
  const startedAt = Date.now();

  /* 1. Validate request shape */
  const parsed = RequestSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({
      ok: false,
      message: "Invalid request: " + parsed.error.errors.map(e => e.message).join("; "),
      warnings: [], assumptions: [], unmapped: [], confidence: 0,
    });
  }
  const { sourceText, hints, appContext } = parsed.data;

  /* 2. Build prompt + user message */
  const systemPrompt = buildSystemPrompt(appContext);
  const userMessage  = buildUserMessage(sourceText, hints);

  /* 3. Call Claude with tool-forced output. */
  let response;
  try {
    const params = {
      model: MODEL,
      max_tokens: MAX_TOKENS,
      system: systemPrompt,
      tools: [submitRecipeDraftTool],
      tool_choice: { type: "tool", name: "submit_recipe_draft" },
      messages: [{ role: "user", content: userMessage }],
    };

    /* Extended thinking improves grid layout substantially.
       Note: when thinking is enabled, tool_choice "tool" is incompatible
       with some configurations, so we use tool_choice "any" then validate. */
    if (ENABLE_THINKING) {
      params.thinking = { type: "enabled", budget_tokens: THINKING_BUDGET };
      params.tool_choice = { type: "any" };
      /* Important: with thinking, max_tokens must exceed budget_tokens */
      if (params.max_tokens <= THINKING_BUDGET + 1000) {
        params.max_tokens = THINKING_BUDGET + 6000;
      }
    }

    response = await anthropic.messages.create(params);
  } catch (err) {
    console.error("[import-pro] Anthropic error:", err?.message || err);
    return res.status(502).json({
      ok: false,
      message: "Upstream model error: " + (err?.message || "unknown"),
      warnings: [], assumptions: [], unmapped: [], confidence: 0,
    });
  }

  /* 4. Extract structured tool input */
  let toolInput = extractToolUse(response);
  if (!toolInput) {
    /* Last-ditch retry: ask the model to call the tool plainly.  This is
       cheap protection against the rare case where the model produced
       text instead of a tool_use block. */
    try {
      const rescue = await anthropic.messages.create({
        model: MODEL,
        max_tokens: MAX_TOKENS,
        system: "You produce only tool calls, never prose.",
        tools: [submitRecipeDraftTool],
        tool_choice: { type: "tool", name: "submit_recipe_draft" },
        messages: [
          { role: "user", content: userMessage },
          { role: "assistant", content: response.content },
          { role: "user", content: "Please call submit_recipe_draft now with your structured draft." },
        ],
      });
      toolInput = extractToolUse(rescue);
    } catch { /* fall through */ }
  }

  if (!toolInput) {
    return res.status(422).json({
      ok: false,
      message: "Model did not return a structured draft.",
      warnings: ["The AI did not call the submit_recipe_draft tool."],
      assumptions: [], unmapped: [], confidence: 0,
    });
  }

  /* 5. Apply our merciless post-processing. */
  let out;
  try {
    out = postProcess(toolInput);
  } catch (err) {
    console.error("[import-pro] post-process error:", err);
    return res.status(500).json({
      ok: false,
      message: "Post-processing failed: " + err.message,
      warnings: [], assumptions: [], unmapped: [], confidence: 0,
    });
  }

  /* 6. Final shape check.  We don't enforce zod here because we already
     guaranteed conformance via tool_use schema + post-processing.  We
     just make sure the contract envelope is intact. */
  const elapsed = Date.now() - startedAt;
  const usage = response.usage || {};
  console.log(`[import-pro] parsed "${out.recipeDraft?.name || "?"}" in ${elapsed}ms ` +
              `(in=${usage.input_tokens}, out=${usage.output_tokens}, model=${MODEL})`);

  /* Always return ok=true if we got a draft, even if the LLM said ok=false
     but produced one anyway. */
  if (out.recipeDraft && out.recipeDraft.preparations?.length) {
    out.ok = true;
  }

  return res.json(out);
});

/*  ─── Health ─── */
app.get("/api/health", (_req, res) => {
  res.json({
    ok:      true,
    service: "import-pro",
    version: "2.0.0",
    model:   MODEL,
    thinking: ENABLE_THINKING,
  });
});

/*  ─── Boot ─── */
app.listen(PORT, () => {
  console.log(`Import Pro v2 backend running on http://localhost:${PORT}`);
  console.log(`Model: ${MODEL}  |  Thinking: ${ENABLE_THINKING ? "on" : "off"}  ` +
              `(budget ${THINKING_BUDGET})`);
  if (!process.env.ANTHROPIC_API_KEY) {
    console.warn("⚠  ANTHROPIC_API_KEY is not set — calls will fail.");
  }
});
