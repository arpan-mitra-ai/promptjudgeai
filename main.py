"""
PromptJudge AI - Backend API
Evaluates and improves AI prompts using LangChain LCEL chains with GPT-4o-mini (configured as gpt-5-mini).
"""

import os
import time
import asyncio
import logging
from typing import Dict, List, Any, Union
from dotenv import load_dotenv

# FastAPI Imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()

# ============================================================================
# LOGGING & SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PromptJudge")

app = FastAPI(
    title="PromptJudge API",
    version="1.0.0",
    description="AI-powered prompt evaluation and improvement service"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class PromptInput(BaseModel):
    """Request model for prompt evaluation"""
    prompt: str = Field(..., min_length=10, max_length=5000, description="The user's AI prompt")

    @field_validator('prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or only whitespace")
        return v.strip()

class CriterionResult(BaseModel):
    """Result for a single evaluation criterion"""
    score: int = Field(..., ge=1, le=10)
    issues: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)

class EvaluationResponse(BaseModel):
    """Complete response with evaluation and improvement"""
    overall_score: float
    criteria: Dict[str, CriterionResult]
    improved_prompt: str
    explanation: str
    processing_time: float

# ============================================================================
# LANGCHAIN LCEL SETUP
# ============================================================================

# Initialize OpenAI LLM
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables!")
    
    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.3,
        max_tokens=2500,
        request_timeout=30,
        api_key=api_key,
        # CRITICAL FIX: Force OpenAI to output valid JSON object
        # This prevents "Invalid json output" errors by enforcing structure at the API level
        model_kwargs={"response_format": {"type": "json_object"}}
    )
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    llm = None

# ----------------------------------------------------------------------------
# CRITICAL FIX: Double curly braces {{ }} used for JSON examples
# Single curly braces { } used ONLY for variables like {prompt}
# ----------------------------------------------------------------------------

evaluation_system_prompt = """
You are PromptJudge, an expert AI prompt evaluator with 10+ years of experience in prompt engineering.

TASK:
Evaluate the provided prompt on 5 criteria: clarity, context, instructions, output_format, constraints.

For each criterion:
1. Assign a score from 1-10 based on the rubrics below
2. Identify 2-4 specific issues by QUOTING exact text from the user's prompt
3. Identify 1-3 strengths by QUOTING exact text from the user's prompt

CRITICAL: Every issue and strength MUST quote the exact text from the prompt. Format: "Quote: 'exact text here' - Problem/Reason: explanation"

Output ONLY raw JSON. Do not use markdown code blocks.

---

SCORING RUBRICS:

CLARITY (Weight: 25%):
9-10: Crystal clear goal, all terms defined, specific action verbs, quantifiable elements, zero ambiguity
7-8: Clear goal, mostly specific language, minor acceptable vagueness, good concrete terms
5-6: Basic goal present, several vague elements, some ambiguity that needs fixing
3-4: Unclear goal, significant ambiguity, many vague terms, poor specificity
1-2: Extremely vague, no clear intent, completely ambiguous

CONTEXT (Weight: 20%):
9-10: Comprehensive context, audience defined, domain knowledge provided, examples included, purpose explicit
7-8: Good context, audience specified, relevant background, clear purpose stated
5-6: Basic context, missing key elements, audience vaguely mentioned, incomplete background
3-4: Minimal context, no audience, missing background, unclear purpose
1-2: No contextual information provided at all

INSTRUCTIONS (Weight: 25%):
9-10: Numbered steps, perfect logical flow, sub-tasks defined, tightly bounded scope, clear workflow
7-8: Structured approach, mostly clear steps, good sequence, reasonable scope definition
5-6: Basic structure present, some organization, scope somewhat defined, acceptable flow
3-4: Poorly structured, unclear steps, confusing sequence, unbounded scope
1-2: No structure, completely chaotic, no organization, no discernible workflow

OUTPUT_FORMAT (Weight: 15%):
9-10: Format, length, structure, style, and presentation all explicitly and clearly specified
7-8: Format and length specified, good structural details, tone guidance provided
5-6: Format mentioned, missing specifications like length or style, incomplete details
3-4: Vague format indication only, missing most critical details
1-2: No output specification whatsoever, completely undefined expectations

CONSTRAINTS (Weight: 15%):
9-10: Clear limitations, explicit exclusions, quality criteria defined, edge cases covered, strong boundaries
7-8: Good constraints, boundaries well-defined, some quality criteria, reasonable limitations
5-6: Basic constraints mentioned, some boundaries present, minimal quality criteria
3-4: Minimal or unclear constraints, weak boundaries, missing quality standards
1-2: No constraints provided at all, completely unbounded

---

CALIBRATION EXAMPLES (Learn from these):

Example 1 - POOR PROMPT (Score ~2):
User Prompt: "Write something about dogs"

Expected Output:
{{
  "clarity": {{
    "score": 2,
    "issues": [
      "Quote: 'something' - Problem: Extremely vague output type, could mean anything from a sentence to a book",
      "Quote: 'about dogs' - Problem: No specific aspect mentioned, topic is far too broad",
      "Quote: 'Write' - Problem: Generic verb with no specificity about what type of writing"
    ],
    "strengths": []
  }},
  "context": {{
    "score": 1,
    "issues": [
      "Missing: No target audience specified",
      "Missing: No purpose or use case mentioned",
      "Missing: No domain knowledge or background provided"
    ],
    "strengths": []
  }},
  "instructions": {{
    "score": 1,
    "issues": [
      "Missing: No steps or structure provided",
      "Missing: No breakdown of what to include",
      "Quote: Single vague request - Problem: No actionable instructions"
    ],
    "strengths": []
  }},
  "output_format": {{
    "score": 1,
    "issues": [
      "Missing: No format specified (article? list? essay?)",
      "Missing: No length indication",
      "Missing: No style or tone requirements"
    ],
    "strengths": []
  }},
  "constraints": {{
    "score": 1,
    "issues": [
      "Missing: No constraints whatsoever",
      "Missing: No limitations or boundaries",
      "Missing: No quality criteria"
    ],
    "strengths": []
  }}
}}

Example 2 - EXCELLENT PROMPT (Score ~9):
User Prompt: "Write a 500-word blog post explaining the benefits of adopting rescue dogs over buying from breeders. Target audience: first-time pet owners aged 25-35 who are considering getting a dog. Include: 1) Cost comparison with specific dollar figures, 2) Health benefits backed by veterinary research, 3) Emotional rewards with real stories. Use a warm, encouraging tone that's informative but not preachy. Format as: engaging introduction with a personal anecdote, three main sections with clear headers, and conclusion with a strong call-to-action linking to local shelters. Avoid technical veterinary jargon and don't mention specific breeds."

Expected Output:
{{
  "clarity": {{
    "score": 9,
    "issues": [],
    "strengths": [
      "Quote: '500-word blog post' - Reason: Specific format and exact length defined",
      "Quote: 'benefits of adopting rescue dogs over buying from breeders' - Reason: Clear comparison goal with specific focus",
      "Quote: 'Include: 1) Cost comparison... 2) Health benefits... 3) Emotional rewards' - Reason: Specific content requirements enumerated"
    ]
  }},
  "context": {{
    "score": 9,
    "issues": [],
    "strengths": [
      "Quote: 'Target audience: first-time pet owners aged 25-35' - Reason: Specific demographic defined",
      "Quote: 'who are considering getting a dog' - Reason: Clear situational context and purpose",
      "Quote: 'with specific dollar figures' and 'backed by veterinary research' - Reason: Domain knowledge requirements specified"
    ]
  }},
  "instructions": {{
    "score": 9,
    "issues": [],
    "strengths": [
      "Quote: 'Include: 1) Cost comparison... 2) Health benefits... 3) Emotional rewards' - Reason: Clear numbered structure with three distinct sections",
      "Quote: 'engaging introduction with a personal anecdote, three main sections with clear headers, and conclusion with a strong call-to-action' - Reason: Step-by-step structure with specific components"
    ]
  }},
  "output_format": {{
    "score": 10,
    "issues": [],
    "strengths": [
      "Quote: '500-word blog post' - Reason: Exact format and length specified",
      "Quote: 'Use a warm, encouraging tone that's informative but not preachy' - Reason: Specific style and tone guidance",
      "Quote: 'Format as: engaging introduction... three main sections... conclusion' - Reason: Detailed structural template provided"
    ]
  }},
  "constraints": {{
    "score": 8,
    "issues": [
      "Missing: Could specify maximum length per section for better balance"
    ],
    "strengths": [
      "Quote: 'Avoid technical veterinary jargon' - Reason: Clear exclusion stated",
      "Quote: 'don't mention specific breeds' - Reason: Explicit constraint on content scope",
      "Quote: 'backed by veterinary research' - Reason: Quality criteria for evidence"
    ]
  }}
}}

Example 3 - MEDIUM PROMPT (Score ~5):
User Prompt: "Explain machine learning to beginners. Make it simple."

Expected Output:
{{
  "clarity": {{
    "score": 6,
    "issues": [
      "Quote: 'Explain' - Problem: Generic verb, doesn't specify depth or approach",
      "Quote: 'Make it simple' - Problem: Vague guideline, no definition of what 'simple' means"
    ],
    "strengths": [
      "Quote: 'machine learning' - Reason: Topic is clearly stated and specific"
    ]
  }},
  "context": {{
    "score": 4,
    "issues": [
      "Quote: 'beginners' - Problem: Undefined audience (tech beginners? complete novices? students?)",
      "Missing: No purpose stated (learning? decision-making? overview?)",
      "Missing: No background on what knowledge to assume"
    ],
    "strengths": []
  }},
  "instructions": {{
    "score": 5,
    "issues": [
      "Missing: No structure for the explanation",
      "Missing: No indication of what aspects to cover",
      "Quote: Single request - Problem: No breakdown of sub-topics"
    ],
    "strengths": [
      "Quote: 'Explain' - Reason: Action is clear even if not detailed"
    ]
  }},
  "output_format": {{
    "score": 3,
    "issues": [
      "Missing: No format specified (article? bullet points? video script?)",
      "Missing: No length indication whatsoever",
      "Quote: 'Make it simple' - Problem: Tone mentioned but style unclear"
    ],
    "strengths": []
  }},
  "constraints": {{
    "score": 4,
    "issues": [
      "Quote: 'simple' - Problem: Vague constraint, needs definition (avoid jargon? use analogies?)",
      "Missing: No length limits or boundaries"
    ],
    "strengths": [
      "Quote: 'to beginners' - Reason: Implies complexity constraint even if not explicit"
    ]
  }}
}}

---

Now evaluate this prompt following the exact JSON format above:
"""

evaluation_template = ChatPromptTemplate.from_messages([
    ("system", evaluation_system_prompt),
    ("human", "{prompt}")
])

if llm:
    evaluation_chain = evaluation_template | llm | JsonOutputParser()
else:
    evaluation_chain = None

# Improvement Templates (Double braces {{ }} for JSON examples)
improvement_system_prompt = """
You are a prompt improvement specialist with expertise in transforming poorly-written prompts into high-quality, production-ready prompts that consistently score 9-10 across all evaluation criteria.
TASK:
Transform the provided prompt into a high-quality version that would score 9-10 on all evaluation criteria (clarity, context, instructions, output_format, constraints).

CRITICAL RULES YOU MUST FOLLOW:
1. PRESERVE INTENT: Keep the core goal unchanged - don't change what the user wants to accomplish
2. FIX ALL ISSUES: Address every single problem identified in the evaluation
3. MAINTAIN VOICE: If the original is casual, keep it casual. If formal, keep it formal. Don't over-formalize friendly prompts.
4. ADD MISSING ELEMENTS: Systematically add context, audience, format, constraints, and structure where missing
5. BE SPECIFIC: Replace all vague language with concrete, specific, actionable terms
6. ADD STRUCTURE: Break complex requests into clear numbered steps with logical flow
7. REALISTIC LENGTH: The improved prompt should be 2-4x longer than the original, but not excessively verbose

YOUR IMPROVEMENTS MUST ADDRESS THESE DIMENSIONS:

CLARITY IMPROVEMENTS:
- Replace generic verbs (write, make, do, create) with specific action verbs (draft, compose, generate, analyze)
- Add quantifiable elements: word counts, number of items, specific measurements
- Define all ambiguous terms explicitly
- Specify the exact deliverable type (blog post, email, code function, analysis report)
- Remove vague qualifiers like "good", "better", "nice", "interesting"

CONTEXT IMPROVEMENTS:
- Add target audience with demographics (age, expertise level, role, background)
- State the purpose or use case explicitly (why is this needed?)
- Provide relevant domain knowledge or background information
- Include examples or reference points when helpful
- Specify the situational context (when, where, how this will be used)

INSTRUCTIONS IMPROVEMENTS:
- Break single requests into numbered steps (1, 2, 3...)
- Add logical sequence and dependencies between steps
- Define clear sub-tasks for complex requests
- Set bounded scope (what to include AND what to exclude)
- Provide a clear workflow from start to finish

OUTPUT_FORMAT IMPROVEMENTS:
- Specify exact format type (essay, list, table, code, JSON, markdown, slides)
- Add precise length constraints (word count, character limit, number of sections)
- Define structure/template (intro, body, conclusion OR header, content, footer)
- Specify style and tone requirements (formal, casual, technical, friendly, professional)
- Add presentation details if relevant (headers, bullet points, code comments)

CONSTRAINTS IMPROVEMENTS:
- Add explicit limitations ("maximum 500 words", "no more than 5 examples")
- State clear exclusions ("do not include...", "avoid mentioning...", "exclude...")
- Define quality criteria ("cite sources", "use peer-reviewed research", "include error handling")
- Address edge cases when relevant ("handle empty inputs", "account for mobile users")
- Set clear boundaries on scope and complexity

EXAMPLES OF GOOD IMPROVEMENTS:

Bad → Good Transformation 1:
Original: "Write about climate change"
Improved: "Write a 1200-word informative article about the economic impacts of climate change on coastal communities, targeted at policymakers and urban planners with basic climate science knowledge. Include: 1) Overview of rising sea levels and their timeline (2020-2050), 2) Case studies from three cities (Miami, Jakarta, Venice) with specific cost data, 3) Economic sectors most affected (real estate, insurance, tourism, infrastructure), 4) Policy recommendations with cost-benefit analysis. Use a professional, evidence-based tone with data from IPCC reports and economic journals. Format as: executive summary (100 words), introduction, four main sections with subheadings, conclusion with actionable recommendations. Cite at least 8 authoritative sources. Avoid political language and focus on economic data."

Bad → Good Transformation 2:
Original: "Create a Python function"
Improved: "Create a Python function named 'process_user_data' that takes a list of user dictionaries as input (each with keys: 'name', 'email', 'age', 'signup_date') and returns a pandas DataFrame sorted by signup_date in descending order. Requirements: 1) Validate that all required keys exist, raising ValueError if missing, 2) Filter out users with invalid email formats (use regex), 3) Convert signup_date strings (format: 'YYYY-MM-DD') to datetime objects, 4) Add a new column 'days_since_signup' calculating days from today, 5) Include comprehensive docstring with parameter types and return value, 6) Add type hints (List[Dict] → pd.DataFrame), 7) Handle edge cases: empty list, None values, duplicate emails (keep first occurrence). Write in PEP 8 style with descriptive variable names. Include 3 example test cases demonstrating normal, edge, and error scenarios."

Bad → Good Transformation 3:
Original: "Explain quantum computing"
Improved: "Explain the fundamental principles of quantum computing to software engineers with no physics background, focusing on practical implications for computing. Cover: 1) Classical bits vs quantum qubits (superposition explained through coin flip analogy), 2) Entanglement concept using paired particles metaphor, 3) Quantum gates vs classical logic gates with visual comparison, 4) Three real-world applications: cryptography (RSA breaking), drug discovery (molecular simulation), optimization (traveling salesman problem), 5) Current limitations and timeline to practical use (2025-2030 outlook). Use conversational tone with analogies to familiar programming concepts. Avoid heavy mathematics - use conceptual explanations instead. Format as 1500-word article with: engaging introduction posing a problem quantum computing solves, five main sections with descriptive headers, conclusion summarizing key takeaways and relevance to software development. Include 2-3 diagrams descriptions for visual clarity. Target reading level: undergraduate computer science."

OUTPUT FORMAT:
You must return valid JSON with exactly these fields:
{{
  "improved_prompt": "The complete enhanced version of the prompt (2-4x longer than original)",
  "changes_made": [
    "Added specific word count: 500 words",
    "Defined target audience: first-time pet owners aged 25-35",
    "Structured content into 3 numbered sections",
    "Specified format: blog post with intro, body, conclusion",
    "Added tone requirement: warm and encouraging"
  ],
  "predicted_score": 9.2,
  "improvement_summary": "Transformed vague request into structured prompt with clear audience, format, and comprehensive content requirements"
}}

IMPORTANT NOTES:
- The improved_prompt should be a complete, standalone prompt that could be used immediately
- Keep the improved_prompt as a single continuous text, not broken into sections
- The changes_made array should list 4-6 most significant improvements (not every tiny change)
- The predicted_score should be 9.0 or higher (if you can't achieve this, you haven't improved enough)
- The improvement_summary should be a single sentence explaining the key transformation

QUALITY CHECKLIST - Your improved prompt must have ALL of these:
✓ Specific deliverable type clearly stated (what exactly to produce)
✓ Target audience explicitly defined (who is this for)
✓ Quantifiable elements present (word count, number of items, etc)
✓ Clear structure or steps (numbered or sequential)
✓ Format and style specified (how it should be written)
✓ Concrete constraints or limitations (boundaries on scope)
✓ No vague terms remaining (all ambiguity removed)

Output ONLY raw JSON. Do not use markdown code blocks.

Now improve this prompt:
"""

improvement_template = ChatPromptTemplate.from_messages([
    ("system", improvement_system_prompt),
    ("human", "Original prompt: {prompt}\nIssues: {issues}\nScores: {scores}")
])

if llm:
    improvement_chain = improvement_template | llm | JsonOutputParser()
else:
    improvement_chain = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def evaluate_and_improve(prompt: str):
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized. Check server logs.")

    # Step 1: Evaluate
    try:
        # We need the full prompt text in the template for this to work well
        eval_result = await evaluation_chain.ainvoke({"prompt": prompt})
    except Exception as e:
        logger.error(f"Evaluation chain failed: {e}")
        # Return a fallback result instead of crashing
        return fallback_evaluation(), fallback_improvement(prompt)

    # Step 2: Prepare data
    all_issues = []
    scores_dict = {}
    
    if isinstance(eval_result, dict):
        for criterion, data in eval_result.items():
            if isinstance(data, dict):
                scores_dict[criterion] = data.get('score', 0)
                issues = data.get('issues', [])
                if issues:
                    all_issues.extend(issues)
    
    # Step 3: Improve
    try:
        improve_result = await improvement_chain.ainvoke({
            "prompt": prompt,
            "issues": "\n".join(all_issues) if all_issues else "Optimize for perfection.",
            "scores": str(scores_dict)
        })
    except Exception as e:
        logger.error(f"Improvement chain failed: {e}")
        improve_result = fallback_improvement(prompt)

    return eval_result, improve_result

def fallback_evaluation():
    """Returns a safe structure if LLM fails"""
    return {
        "clarity": {"score": 5, "issues": ["Analysis failed"], "strengths": []},
        "context": {"score": 5, "issues": ["Analysis failed"], "strengths": []},
        "instructions": {"score": 5, "issues": ["Analysis failed"], "strengths": []},
        "output_format": {"score": 5, "issues": ["Analysis failed"], "strengths": []},
        "constraints": {"score": 5, "issues": ["Analysis failed"], "strengths": []}
    }

def fallback_improvement(prompt):
    return {
        "improved_prompt": prompt,
        "changes_made": ["System error prevented improvement"],
        "improvement_summary": "Original retained."
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate_prompt(request: PromptInput):
    start_time = time.time()
    logger.info(f"Received request: {request.prompt[:50]}...")

    try:
        # Increased timeout to 45s for cold starts
        eval_result, improve_result = await asyncio.wait_for(
            evaluate_and_improve(request.prompt), 
            timeout=45.0
        )

        # Calculate Score
        weights = {"clarity": 0.25, "context": 0.20, "instructions": 0.25, "output_format": 0.15, "constraints": 0.15}
        overall_score = 0.0
        total_issues = 0
        
        for criterion, weight in weights.items():
            if criterion in eval_result:
                score = eval_result[criterion].get("score", 0)
                overall_score += score * weight
                total_issues += len(eval_result[criterion].get("issues", []))
            else:
                overall_score += 5 * weight 

        overall_score = round(overall_score, 1)

        # Explanation
        changes = improve_result.get("changes_made", [])
        changes_str = "general improvements"
        if isinstance(changes, list) and changes:
            # Handle string or dict changes safely
            top_changes = changes[:3]
            formatted_changes = []
            for c in top_changes:
                if isinstance(c, dict):
                    formatted_changes.append(str(c.get("change", "")))
                else:
                    formatted_changes.append(str(c))
            changes_str = ", ".join(formatted_changes)

        explanation = f"Score: {overall_score}/10. addressed {total_issues} issues: {changes_str}."
        processing_time = round(time.time() - start_time, 2)
        
        return EvaluationResponse(
            overall_score=overall_score,
            criteria=eval_result,
            improved_prompt=improve_result.get("improved_prompt", request.prompt),
            explanation=explanation,
            processing_time=processing_time
        )

    except asyncio.TimeoutError:
        logger.error("Timeout")
        raise HTTPException(status_code=504, detail="AI timeout. Please try again.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "PromptJudge AI"}

@app.get("/")
async def root():
    return {"message": "PromptJudge AI API is running"}
