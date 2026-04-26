import { config } from "dotenv";
config(); // Load .env file — must be before any process.env reads
 
import express from "express";
import cors from "cors";
import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";
 
/* ═══════════════════════════════════════════════════════════════════
   Mise en Code — Import Pro Backend
   ─────────────────────────────────────────────────────────────────
   POST /api/import-pro/parse
   
   Receives recipe text + app context, calls Claude, returns
   a structured recipeDraft for preview and import.
═══════════════════════════════════════════════════════════════════ */
 
const PORT = process.env.PORT || 3001;
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || "*").split(",").map(s => s.trim());
 
const app = express();
app.use(express.json({ limit: "500kb" }));
app.use(cors({
  origin: ALLOWED_ORIGINS[0] === "*" ? true : ALLOWED_ORIGINS,
  methods: ["POST", "OPTIONS"],
}));
 
/* ── Anthropic client ── */
const anthropic = new Anthropic(); // reads ANTHROPIC_API_KEY from env
 
/* ── Request validation ── */
const RequestSchema = z.object({
  sourceText: z.string().min(10).max(50000),
  hints: z.object({
    title: z.string().optional(),
    servings: z.number().int().min(1).max(100).optional(),
    language: z.enum(["auto", "en", "es", "fr"]).default("auto"),
    importMode: z.enum(["smart", "preserve", "professional"]).default("smart"),
  }),
  appContext: z.object({
    ingredientLibrary: z.array(z.object({ id: z.string(), n: z.string(), cat: z.string().optional() })).optional(),
    toolLibrary: z.array(z.object({ id: z.string(), n: z.string() })).optional(),
    processLibrary: z.array(z.string()).optional(),
    customIngredients: z.any().optional(),
    customTools: z.any().optional(),
    customProcesses: z.any().optional(),
    schemaVersion: z.number().default(1),
  }),
});
 
/* ── System prompt for the Import Pro agent ── */
const buildSystemPrompt = (ctx) => {
  const ingList = (ctx.ingredientLibrary || []).map(i => i.id).join(", ");
  const toolList = (ctx.toolLibrary || []).map(t => t.id).join(", ");
  const procList = (ctx.processLibrary || []).join(", ");
  const customIngs = ctx.customIngredients ? JSON.stringify(ctx.customIngredients) : "{}";
  const customTools = ctx.customTools ? JSON.stringify(ctx.customTools) : "{}";
 
  return `You are Import Pro, a culinary recipe-structuring AI agent for the Mise en Code recipe builder.
 
Your job: convert natural-language recipes into the app's exact JSON schema.
 
CRITICAL RULES:
1. Output ONLY valid JSON. No markdown. No prose. No backticks.
2. Use ONLY canonical IDs from the provided libraries.
3. If an ingredient/tool/process is NOT in the library, add it to the appropriate "candidates" list — never invent a canonical ID.
4. Preserve uncertainty in warnings and assumptions.
5. Never silently discard recipe information.
6. Return the EXACT top-level shape specified below.
 
═══ APP SCHEMA ═══
 
Recipe: { id, name, description, servings, status:"draft", tags:[], category, flavour:{sweetness,richness,spiciness,acidity,umami}, preparations:[] }
Preparation: { id, name, comment, isModule:false, laneNames:[], grid:[][] }
Step (or null): { id, title, instruction, ingredients:[], processes:[], tool:null|{uid,toolId,name,opts:{}}, processParams:{}, connectsTo:[] }
StepIngredient: { uid, ingredientId, name, qty, unit, isEgg?, eggPart?, isChilli?, chilliHeat? }
 
═══ CANONICAL LIBRARIES ═══
 
INGREDIENTS: ${ingList}
TOOLS: ${toolList}
PROCESSES: ${procList}
CUSTOM INGREDIENTS: ${customIngs}
CUSTOM TOOLS: ${customTools}
 
═══ TOOL OPTS PATTERNS ═══
- oven: { "_temp": 180, "_time": 30, "_timeUnit": "min", "ovenMode": "Conventional" }
- thermomix: { "_temp": 80, "_speed": "3", "_time": 10, "_timeUnit": "min" }
- kitchenaid: { "_speed": "6", "attachment": "Wire Whip", "_time": 5, "_timeUnit": "min" }
- cast-iron-pan / wok / reg-pot / stock-pot / saucepan: { "heat": "Medium-High" }
  Heat options: Off, Low, Medium-Low, Medium, Medium-High, High
- knife: { "cut": "Dice" }
  Cut options: Brunoise, Small Dice, Medium Dice, Batonnet, Julienne, Chiffonade, Slice, Mince
- Other tools: {} empty opts
 
═══ UNITS ═══
g, kg, ml, l, tsp, tbsp, cup, ea, pinch, bunch, clove, slice, strip, sprig, to taste
 
═══ INFERENCE RULES ═══
 
INGREDIENT MAPPING: Match to canonical IDs. "butter" → "butter". "olive oil" → "olive-oil". If no match exists, DO NOT use the canonical list — add to customIngredientCandidates with: { name, category, unit, reason, confidence }.
 
TOOL INFERENCE: "bake"→oven, "whisk"→whisk or kitchenaid, "fry/sear/sauté"→cast-iron-pan or wok, "simmer/boil"→reg-pot, "chop/dice"→knife. If uncertain, use the simplest tool and add a warning.
 
PROCESS MAPPING: Use exact keys from the process library. If a technique has no match, use the closest approved process and describe the actual technique in the step instruction. Add to customProcessCandidates if truly novel.
 
LANE RULES:
- Each ingredient group heading ("Beef", "For the sauce") = a lane
- grid[row][lane], rows = time sequence, null = empty cell
- Place ingredients ONLY at first use — later steps inherit them
- connectsTo when output physically moves between lanes
- 1–5 lanes typical. Simple recipes = 1 lane.
- Vegetables cooked WITH protein = same lane
- "Meanwhile" / "While X cooks" = separate lanes
- Timeline ("60 min before serving") = row ordering
 
CONNECTIONS: Use connectsTo only when output PHYSICALLY MOVES between lanes (e.g., "drain pasta into the pan", "pour syrup over egg whites").
 
FLAVOUR ESTIMATION: Estimate 0–100 for each axis. Be conservative. The app recomputes live, so this is just an initial estimate.
 
IDS: Generate random 8-character alphanumeric IDs for all id/uid fields. Make them unique.
 
═══ REQUIRED OUTPUT SHAPE ═══
 
{
  "ok": true,
  "recipeDraft": { <full Recipe object> },
  "customIngredientCandidates": [
    { "name": "...", "category": "...", "unit": "...", "reason": "...", "confidence": 0.8 }
  ],
  "customToolCandidates": [
    { "name": "...", "category": "...", "reason": "...", "confidence": 0.8 }
  ],
  "customProcessCandidates": [
    { "name": "...", "category": "...", "reason": "...", "confidence": 0.8 }
  ],
  "warnings": ["..."],
  "assumptions": ["..."],
  "unmapped": ["..."],
  "confidence": 0.85
}
 
If the recipe cannot be reliably parsed:
{
  "ok": false,
  "message": "...",
  "warnings": [],
  "assumptions": [],
  "unmapped": [],
  "partialDraft": null,
  "confidence": 0.0
}`;
};
 
/* ── Build user message from request ── */
const buildUserMessage = (sourceText, hints) => {
  const parts = [];
  if (hints.title) parts.push(`RECIPE NAME: ${hints.title}`);
  if (hints.servings) parts.push(`SERVINGS: ${hints.servings}`);
  if (hints.language && hints.language !== "auto") parts.push(`LANGUAGE: ${hints.language}`);
  parts.push(`IMPORT MODE: ${hints.importMode}`);
  parts.push("");
  parts.push("RECIPE TEXT:");
  parts.push(sourceText);
  return parts.join("\n");
};
 
/* ── Response validation ── */
const RecipeDraftSchema = z.object({
  ok: z.boolean(),
  recipeDraft: z.any().optional(),
  customIngredientCandidates: z.array(z.any()).default([]),
  customToolCandidates: z.array(z.any()).default([]),
  customProcessCandidates: z.array(z.any()).default([]),
  warnings: z.array(z.string()).default([]),
  assumptions: z.array(z.string()).default([]),
  unmapped: z.array(z.string()).default([]),
  confidence: z.number().min(0).max(1).default(0),
  message: z.string().optional(),
  partialDraft: z.any().optional(),
});
 
/* ── Main endpoint ── */
app.post("/api/import-pro/parse", async (req, res) => {
  try {
    /* Validate request */
    const parsed = RequestSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({
        ok: false,
        message: "Invalid request: " + parsed.error.errors.map(e => e.message).join(", "),
        warnings: [],
        assumptions: [],
        unmapped: [],
        confidence: 0,
      });
    }
 
    const { sourceText, hints, appContext } = parsed.data;
 
    /* Call Claude */
    const systemPrompt = buildSystemPrompt(appContext);
    const userMessage = buildUserMessage(sourceText, hints);
 
    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 8000,
      system: systemPrompt,
      messages: [{ role: "user", content: userMessage }],
    });
 
    /* Extract text response */
    const rawText = response.content
      .filter(b => b.type === "text")
      .map(b => b.text)
      .join("");
 
    /* Clean potential markdown fences */
    const cleaned = rawText
      .replace(/^```json\s*/i, "")
      .replace(/^```\s*/i, "")
      .replace(/\s*```$/, "")
      .trim();
 
    /* Parse JSON */
    let result;
    try {
      result = JSON.parse(cleaned);
    } catch (parseErr) {
      return res.status(422).json({
        ok: false,
        message: "Claude returned invalid JSON: " + parseErr.message,
        warnings: ["The AI response could not be parsed as JSON."],
        assumptions: [],
        unmapped: [],
        confidence: 0,
        rawPreview: cleaned.substring(0, 500),
      });
    }
 
    /* Validate response shape */
    const validated = RecipeDraftSchema.safeParse(result);
    if (!validated.success) {
      return res.status(422).json({
        ok: false,
        message: "Response shape invalid: " + validated.error.errors.map(e => e.message).join(", "),
        warnings: [],
        assumptions: [],
        unmapped: [],
        confidence: 0,
      });
    }
 
    /* Ensure draft has status:"draft" */
    if (validated.data.ok && validated.data.recipeDraft) {
      validated.data.recipeDraft.status = "draft";
    }
 
    return res.json(validated.data);
 
  } catch (err) {
    console.error("Import Pro error:", err);
    return res.status(500).json({
      ok: false,
      message: "Server error: " + (err.message || "Unknown"),
      warnings: [],
      assumptions: [],
      unmapped: [],
      confidence: 0,
    });
  }
});
 
/* ── Health check ── */
app.get("/api/health", (req, res) => {
  res.json({ ok: true, service: "import-pro", version: "1.0.0" });
});
 
/* ── Start server ── */
app.listen(PORT, () => {
  console.log(`Import Pro backend running on http://localhost:${PORT}`);
  console.log(`Endpoint: POST http://localhost:${PORT}/api/import-pro/parse`);
  if (!process.env.ANTHROPIC_API_KEY) {
    console.warn("⚠ ANTHROPIC_API_KEY not set — API calls will fail");
  }
});