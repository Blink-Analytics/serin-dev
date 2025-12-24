var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
require('dotenv').config();
var xlsx = require('xlsx');
var path = require('path');
var Groq = require('groq-sdk');
var groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
function main() {
    return __awaiter(this, void 0, void 0, function () {
        // Helper to parse JSON fields safely
        function tryParseJSON(val) {
            if (typeof val !== 'string')
                return val;
            try {
                return JSON.parse(val);
            }
            catch (e) {
                return val;
            }
        }
        // V4 PRODUCTION PROMPT (DO NOT CHANGE)
        function generateInterviewScoringPromptV4(jobDescriptionJson, objectivesList) {
            var objectivesFormatted = objectivesList.map(function (obj, i) { return "".concat(i + 1, ". ").concat(obj); }).join('\n');
            return "### PRODUCTION SCORING SYSTEM V4 ###\n\n**YOUR IDENTITY:**\nYou are a Senior Technical Hiring Manager with 15+ years conducting 10,000+ technical interviews across FAANG, startups, and enterprises. You are known for being BRUTALLY HONEST, data-driven, and having a 95% interview-to-hire accuracy rate.\n\n**YOUR MANDATE:**\nScore ALL ".concat(objectivesList.length, " objectives for this job by comparing them DIRECTLY to each other and to industry standards. Your scores determine which questions get asked first and how deeply we probe\u2014bad scoring means bad hires costing $200K+ per mistake.\n\n**THE STAKES:**\n- Over-scoring (giving everything 8-9) = Weak interviews, bad hires, company failure\n- Under-differentiation = Wasting time on low-value topics, missing critical skills\n- Your scores DIRECTLY impact whether we hire the right person\n---\n\n**JOB CONTEXT:**\n").concat(jobDescriptionJson, "\n\n**OBJECTIVES TO SCORE (").concat(objectivesList.length, " total):**\n").concat(objectivesFormatted, "\n\n---\n**\u26A0\uFE0F CRITICAL: DO NOT ASSUME THE INPUT ORDER HAS MEANING! \u26A0\uFE0F**\n\nThe objectives above are listed in an ARBITRARY order. They:\n- **MIGHT** be randomly shuffled\n- **MIGHT** be pre-sorted by importance (but probably aren't)\n- **MIGHT** be in reverse order\n- **MIGHT** be grouped by category\n- **The list position tells you NOTHING about importance or difficulty**\n\n**YOUR JOB:** Ignore the numbers (1, 2, 3, 4, 5). Read what each objective SAYS and score based on that content alone.\n\n\u274C **WRONG APPROACH (DO NOT DO THIS):**\n\"Objective #1 is first, so I'll give it 9.5\"\n\"Objective #2 is second, so I'll give it 7.8\"\n\"I'll just assign decreasing scores: 9\u21928\u21927\u21926\u21925\"\n\n\u2705 **CORRECT APPROACH:**\n\"Objective #1 says 'Update Jira tickets'\u2014that's low priority admin work \u2192 2.0\"\n\"Objective #2 says 'Design distributed ML pipeline'\u2014that's the CORE job function \u2192 9.5\"\n\"Objective #3 says 'Attend team meetings'\u2014routine, not critical \u2192 3.5\"\n\"Objective #4 says 'Optimize database for 100k QPS'\u2014high technical impact \u2192 8.5\"\n\"Objective #5 says 'Fix minor CSS bugs'\u2014trivial work \u2192 1.5\"\n\n**RESULT:** Scores are [2.0, 9.5, 3.5, 8.5, 1.5] \u2014 NOT in decreasing order!\n\n**REAL-WORLD EXAMPLE:**\n\nIf you see these objectives:\n1. \"Production-Grade Model Deployment & Optimization\"\n2. \"MLOps Lifecycle Management & Automation\"\n3. \"Deep Learning & Algorithmic Problem Solving\"\n4. \"Cross-Functional Technical Communication\"\n5. \"Infrastructure & Cost Efficiency Architecture\"\n\nYou might think: \"These look ranked already, I'll score 9\u21928\u21927\u21926\u21925\"\n\n\u274C WRONG! Read the CONTENT:\n- #1, #2, #3, #5 are ALL core ML Engineering skills (should be 8.5-9.5)\n- #4 is important but secondary (maybe 6.5-7.5)\n- You need to differentiate WITHIN the technical ones based on job specifics\n\nCorrect scores might be: [9.2, 8.5, 9.5, 6.8, 8.8] \u2014 NOT sequential!\n\n---\n**CRITICAL SCORING RULES (FOLLOW OR YOUR OUTPUT IS REJECTED):**\n\n**RULE 1: FORCED DISTRIBUTION**\nFor ").concat(objectivesList.length, " objectives, you MUST create this distribution:\n- **Exactly 1-2 objectives**: 9.0-10.0 (the MOST critical skills for this role)\n- **~20% of objectives**: 7.0-8.5 (important but not make-or-break)\n- **~50% of objectives**: 4.0-6.5 (standard/expected at this level)\n- **~20% of objectives**: 2.0-3.5 (nice-to-have or peripheral)\n- **At least 1 objective**: 1.0-2.0 (low priority/tangential)\n\n**RULE 2: THE COMPARISON METHOD (CONTENT-BASED, NOT POSITION-BASED)**\nYou CANNOT score based on list position. Use this process:\n1. Read all ").concat(objectivesList.length, " objectives first (IGNORE the numbering 1, 2, 3, ...)\n2. Ask: \"Which ONE objective describes the most critical task for success in this role?\" \u2192 That's your 9-10 anchor (could be objective #1, #3, #5, or ANY position)\n3. Ask: \"Which objective describes the most peripheral/mundane task?\" \u2192 That's your 1-3 anchor (could be #2, #4, or ANY position)\n4. Rank the rest BETWEEN these anchors based on CONTENT, not list order\n5. Check: Do I have proper spread? (highest - lowest \u2265 6 points)\n6. **VERIFY:** Are my scores based on what each objective SAYS, not where it appears in the list?\n7. **ANTI-PATTERN CHECK:** Are my scores NOT in perfect decreasing order (9.2\u21927.8\u21925.5\u21923.8\u21922.2)?\n\n**RULE 3: THE INTERVIEW TIME TEST**\nFor importance scoring, use this calibration:\n- **9.0-10.0**: \"I would dedicate 20+ minutes to this in a 60-min interview\"\n- **7.0-8.5**: \"I would spend 10-15 minutes on this\"\n- **5.0-6.5**: \"I would spend 5-10 minutes on this\"\n- **3.0-4.5**: \"I would spend 3-5 minutes on this\"\n- **1.0-2.5**: \"I would skip this if pressed for time\"\n\n**RULE 4: THE REJECTION TEST**\nBefore finalizing importance, ask:\n- \"If the candidate ONLY failed this one objective, would I reject them?\"\n  - YES, absolutely \u2192 9-10\n  - Probably yes \u2192 7-8\n  - Depends on other factors \u2192 5-6\n  - Probably no \u2192 3-4\n  - Definitely no \u2192 1-2\n\n**RULE 5: DECIMAL PRECISION**\n- Use decimals (7.5, 8.3, etc.) to show nuanced ranking\n- Never give identical scores unless objectives are truly identical in importance/difficulty\n- Differentiate even within tiers (e.g., 7.2 vs 7.8)\n\n---\n\n**IMPORTANCE SCALE (Interview Priority):**\n\n**9.0-10.0 (DEAL-BREAKER - Reserve for 1-2 objectives max):**\n- If they fail THIS, I will NOT hire them, regardless of other strengths\n- This is THE primary reason this role exists\n- I would spend 20+ minutes of a 60-minute interview on this alone\n- Example: \"Build ML models\" for ML Engineer role\n\n**7.0-8.5 (MUST-HAVE - ~20% of objectives):**\n- Critical for success, but not the entire role\n- Strong performance here indicates a qualified candidate\n- I would spend 10-15 minutes on this\n- Example: \"Write unit tests\" for Senior Backend Engineer\n\n**5.0-6.5 (STANDARD/EXPECTED - ~50% of objectives should fall here):**\n- Part of the job, but routine/expected at this level\n- Not a differentiator between candidates\n- I would spend 5-10 minutes on this\n- Example: \"Participate in code reviews\" for Mid-Level Engineer\n\n**3.0-4.5 (NICE-TO-HAVE - ~20% of objectives):**\n- Beneficial but not required\n- Can be trained post-hire\n- I would spend 3-5 minutes on this\n- Example: \"Write technical documentation\" for Junior Developer\n\n**1.0-2.5 (LOW PRIORITY - At least 1 objective should be here):**\n- Tangential or barely related to core function\n- Only ask if we have extra time\n- Would skip if pressed for time\n- Example: \"Update Jira tickets\" for Senior Architect\n\n**0.0 (IRRELEVANT):**\n- Completely unrelated to the job\n\n---\n\n**DIFFICULTY SCALE (Absolute Task Complexity):**\n\n**9.0-10.0 (CUTTING-EDGE - Extremely rare):**\n- Requires inventing new solutions or pushing industry boundaries\n- <1% of experts in the field can do it\n- 10+ years of specialized mastery required\n- Example: \"Design a novel distributed consensus algorithm\"\n\n**7.0-8.5 (DEEP EXPERTISE - ~10-15% of objectives):**\n- Requires 5-7+ years of specialized experience\n- Complex system design or optimization\n- Requires mastery of advanced concepts\n- Example: \"Optimize database for 100k writes/sec\"\n\n**5.0-6.5 (INTERMEDIATE - ~40-50% of objectives):**\n- Standard implementation with moderate complexity\n- Requires 2-4 years of experience\n- Following best practices and patterns\n- Example: \"Build REST API with authentication\"\n\n**3.0-4.5 (BASIC - ~30% of objectives):**\n- Routine work any competent developer can do\n- Following tutorials/documentation\n- 0-2 years of experience\n- Example: \"Fix bugs in backlog\"\n\n**1.0-2.5 (TRIVIAL - ~10% of objectives):**\n- Can be done in hours by a beginner\n- No specialized knowledge required\n- Example: \"Update CSS color values\"\n\n---\n\n**QUALITY CONTROL CHECKLIST (Run before outputting):**\n\nBefore returning your JSON, verify:\n1. \u2705 Do I have exactly 1-2 objectives with importance 9-10?\n2. \u2705 Is my spread (highest - lowest) \u2265 6 points?\n3. \u2705 Are ~50% of objectives in the 4-7 range?\n4. \u2705 Do I have at least 1 objective below 3.0?\n5. \u2705 Did I use decimals to differentiate (no clustering at 8.0)?\n6. \u2705 Would a real interviewer agree with my top 3 priorities?\n7. \u2705 **CRITICAL: Are my scores in RANDOM/VARIED order, NOT perfectly decreasing (9.2\u21927.8\u21925.5\u21923.8\u21922.2)?**\n8. \u2705 **Did I score based on CONTENT (what the text says), not POSITION (where it appears in the list)?**\n9. \u2705 **If I sorted my scores from high to low, does the order make sense for THIS specific job?**\n\nIf ANY answer is NO, revise your scores before outputting.\n\n**ANTI-PATTERN DETECTION:**\nIf your importance scores are: [9.2, 7.8, 5.5, 3.8, 2.2] or [8.5, 7.2, 6.0, 4.2, 2.8]\n\u2192 YOU FAILED. You scored by position, not content. Start over.\n\nValid patterns look like: [7.2, 9.5, 3.0, 8.5, 2.2] or [5.5, 6.8, 9.2, 3.5, 7.8]\n\u2192 These show you evaluated each objective's actual content.\n\n---\n\n**OUTPUT FORMAT:**\nReturn ONLY a JSON array with one object per objective (same order as input, but values vary based on content):\n\nExample with 5 objectives:\n[\n  {\"importance\": 5.2, \"difficulty\": 4.0},\n  {\"importance\": 9.5, \"difficulty\": 8.0},\n  {\"importance\": 2.0, \"difficulty\": 2.5},\n  {\"importance\": 7.8, \"difficulty\": 6.5},\n  {\"importance\": 3.5, \"difficulty\": 3.0}\n]\n\n**NOTICE:** The scores jump around! Objective #2 scored highest (9.5), not #1.\nThis is because #2's CONTENT was most critical, not because of its position.\n\n**VALIDATION:**\n- Array length MUST equal ").concat(objectivesList.length, "\n- Each object MUST have exactly 2 keys: \"importance\" and \"difficulty\"\n- Values MUST be numbers (decimals preferred, e.g., 7.5 not 7 or \"high\")\n- Range: 0.0 \u2264 value \u2264 10.0\n- NO markdown code blocks (```json), NO explanatory text, NO reasoning in output\n- **Scores should NOT be in perfect decreasing order (that proves you didn't read the content!)**\n\n---\n\n**REMEMBER:**\n- You are scoring for a REAL interview with REAL consequences\n- Over-generous scoring = bad hires = company failure\n- Use the FULL 1-10 scale\u2014most objectives should be 3-7, NOT 8-9\n- When in doubt between two scores, choose the LOWER one\n- Your reputation as a hiring manager depends on accurate differentiation\n- **READ WHAT EACH OBJECTIVE SAYS. The list order is ARBITRARY.**\n- **Even if objectives LOOK pre-sorted, you must validate by reading the content**\n- **Your scores will vary in unpredictable patterns because real importance varies unpredictably**\n\n**NOW SCORE. BE RUTHLESS. BE PRACTICAL. BE ACCURATE. READ THE CONTENT, NOT THE POSITION.**\n");
        }
        var excelPath, workbook, sheetName, worksheet, jsonData, firstRow, extracted, jobData, jobDescriptionJson, objectivesList, prompt, chatCompletion;
        var _a, _b;
        return __generator(this, function (_c) {
            switch (_c.label) {
                case 0:
                    excelPath = path.resolve(__dirname, '../datasets/batched_objectives.xlsx');
                    workbook = xlsx.readFile(excelPath);
                    sheetName = workbook.SheetNames[0];
                    worksheet = workbook.Sheets[sheetName];
                    jsonData = xlsx.utils.sheet_to_json(worksheet, { defval: '' });
                    firstRow = jsonData[0];
                    extracted = {
                        Core_details: tryParseJSON(firstRow.Core_Details || ''),
                        responsibilities: tryParseJSON(firstRow.Responsibilities || ''),
                        experience: tryParseJSON(firstRow.Experience || ''),
                        objectives: firstRow.Objective || []
                    };
                    jobData = extracted;
                    console.log('Extracted job data:', JSON.stringify(jobData, null, 2));
                    jobDescriptionJson = JSON.stringify({
                        Core_details: jobData.Core_details,
                        responsibilities: jobData.responsibilities,
                        experience: jobData.experience
                    }, null, 2);
                    objectivesList = Array.isArray(jobData.objectives) ? jobData.objectives : [];
                    prompt = generateInterviewScoringPromptV4(jobDescriptionJson, objectivesList);
                    return [4 /*yield*/, groq.chat.completions.create({
                            messages: [
                                {
                                    role: "user",
                                    content: prompt,
                                },
                            ],
                            model: "llama-3.1-8b-instant",
                        })];
                case 1:
                    chatCompletion = _c.sent();
                    console.log('Groq API response:', ((_b = (_a = chatCompletion.choices[0]) === null || _a === void 0 ? void 0 : _a.message) === null || _b === void 0 ? void 0 : _b.content) || "");
                    return [2 /*return*/];
            }
        });
    });
}
main().catch(console.error);
