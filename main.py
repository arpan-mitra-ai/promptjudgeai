"""
PromptJudge AI - Backend API
Evaluates and improves AI prompts using LangChain LCEL chains with GPT-4o-mini.
"""

import os
import time
import asyncio
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv

# FastAPI Imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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

# CORS Configuration (Allow all for mobile app access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class PromptInput(BaseModel):
    """Request model for prompt evaluation"""
    prompt: str = Field(..., min_length=10, max_length=5000, description="The user's AI prompt")

    @field_validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
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

# 1. LLM Initialization
# Using temperature 0.3 for a balance of creativity (improvements) and consistency (scoring)
try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=2500,
        request_timeout=30,
        api_key=os.getenv("OPENAI_API_KEY")
    )
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise RuntimeError("LLM Configuration Error")

# 2. Evaluation Chain Definition
evaluation_system_prompt = """You are PromptJudge, an expert AI prompt evaluator with 10+ years of experience in prompt engineering for GPT, Claude, and other LLMs. Your evaluations are known for being thorough, specific, and actionable.

TASK:
Evaluate the provided prompt on 5 criteria: clarity, context, instructions, output_format, constraints.

For each criterion:
1. Assign a score from 1-10 based on the rubrics below
2. Identify 2-4 specific issues by QUOTING exact text from the user's prompt
3. Identify 1-3 strengths by QUOTING exact text from the user's prompt

CRITICAL: Every issue and strength MUST quote the exact text from the prompt. Format: "Quote: 'exact text here' - Problem/Reason: explanation"

Output ONLY valid JSON. No markdown code blocks, no explanations outside JSON, no preamble.

---
SCORING RUBRICS:

CLARITY (Weight: 25%):
9-10: Crystal clear goal, all terms defined, specific action verbs, quantifiable elements, zero ambiguity
7-8: Clear goal, mostly specific language, minor acceptable vagueness, good concrete terms
5-6: Basic goal present, several vague elements, some ambiguity that needs fixing
3-4: Unclear goal, significant ambiguity, many vague terms, poor specificity
1-2: Extremely vague, no clear intent, completely ambiguous, fundamentally unclear

CONTEXT (Weight: 20%):
9-10: Comprehensive context, audience defined, domain knowledge provided, examples included, purpose explicit
7-8: Good context, audience specified, relevant background, clear purpose stated
5-6: Basic context, missing key elements, audience vaguely mentioned, incomplete background
3-4: Minimal context, no audience, missing background, unclear purpose
1-2: No contextual information provided at all, complete absence of framing

INSTRUCTIONS (Weight: 25%):
9-10: Numbered steps, perfect logical flow, sub-tasks defined, tightly bounded scope, clear workflow
7-8: Structured approach, mostly clear steps, good sequence, reasonable scope definition
5-6: Basic structure present, some organization, scope somewhat defined, acceptable flow
3-4: Poorly structured, unclear steps, confusing sequence, unbounded scope, weak organization
1-2: No structure, completely chaotic, no organization, no discernible workflow

OUTPUT_FORMAT (Weight: 15%):
9-10: Format, length, structure, style, and presentation all explicitly and clearly specified
7-8: Format and length specified, good structural details, tone guidance provided
5-6: Format mentioned, missing specifications like length or style, incomplete details
3-4: Vague format indication only, missing most critical details, poor specification
1-2: No output specification whatsoever, completely undefined expectations

CONSTRAINTS (Weight: 15%):
9-10: Clear limitations, explicit exclusions, quality criteria defined, edge cases covered, strong boundaries
7-8: Good constraints, boundaries well-defined, some quality criteria, reasonable limitations
5-6: Basic constraints mentioned, some boundaries present, minimal quality criteria
3-4: Minimal or unclear constraints, weak boundaries, missing quality standards
1-2: No constraints provided at all, completely unbounded, no guardrails

---
CALIBRATION EXAMPLES (Learn from these):

Example 1 - POOR PROMPT:
User Prompt: "Write something about dogs"
Expected Output:
{{
  "clarity": {{
    "score": 2,
    "issues": [
      "Quote: 'something' - Problem: Extremely vague output type, could mean anything from a sentence to a book",
      "Quote: 'about dogs' - Problem: No specific aspect mentioned, topic is far too broad"
    ],
    "strengths": []
  }},
  "context": {{
    "score": 1,
    "issues": [
      "Missing: No target audience specified",
      "Missing: No purpose or use case mentioned"
    ],
    "strengths": []
  }},
  "instructions": {{
    "score": 1,
    "issues": [
      "Missing: No steps or structure provided",
      "Quote: Single vague request - Problem: No actionable instructions"
    ],
    "strengths": []
  }},
  "output_format": {{
    "score": 1,
    "issues": [
      "Missing: No format specified (article? list? essay?)",
      "Missing: No length indication"
    ],
    "strengths": []
  }},
  "constraints": {{
    "score": 1,
    "issues": [
      "Missing: No constraints whatsoever"
    ],
    "strengths": []
  }}
}}

Example 2 - EXCELLENT PROMPT:
User Prompt: "Write a 500-word blog post explaining the benefits of adopting rescue dogs. Target audience: first-time pet owners aged 25-35. Include: 1) Cost comparison, 2) Health benefits. Use a warm, encouraging tone. Format as: intro, three main sections, conclusion. Do not mention specific breeds."
Expected Output:
{{
  "clarity": {{
    "score": 9,
    "issues": [],
    "strengths": [
      "Quote: '500-word blog post' - Reason: Specific format and exact length defined",
      "Quote: 'benefits of adopting rescue dogs' - Reason: Clear goal with specific focus"
    ]
  }},
  "context": {{
    "score": 9,
    "issues": [],
    "strengths": [
      "Quote: 'Target audience: first-time pet owners aged 25-35' - Reason: Specific demographic defined"
    ]
  }},
  "instructions": {{
    "score": 9,
    "issues": [],
    "strengths": [
      "Quote: 'Include: 1) Cost comparison, 2) Health benefits' - Reason: Clear numbered structure"
    ]
  }},
  "output_format": {{
    "score": 10,
    "issues": [],
    "strengths": [
      "Quote: 'Format as: intro, three main sections, conclusion' - Reason: Detailed structural template provided",
      "Quote: 'warm, encouraging tone' - Reason: Specific style guidance"
    ]
  }},
  "constraints": {{
    "score": 9,
    "issues": [],
    "strengths": [
      "Quote: 'Do not mention specific breeds' - Reason: Explicit exclusion constraint"
    ]
  }}
}}
"""

evaluation_template = ChatPromptTemplate.from_messages([
    ("system", evaluation_system_prompt),
    ("human", "{prompt}")
])

# LCEL: Prompt -> LLM -> JSON Parser
evaluation_chain = (
    evaluation_template 
    | llm 
    | JsonOutputParser()
)

# 3. Improvement Chain Definition
improvement_system_prompt = """You are a prompt improvement specialist with expertise in transforming poorly-written prompts into high-quality, production-ready prompts.

TASK:
Transform the provided prompt into a high-quality version that would score 9-10 on all evaluation criteria.

RULES:
1. PRESERVE INTENT: Keep the core goal unchanged - don't change what the user wants
2. FIX ALL ISSUES: Address every problem identified in the evaluation
3. MAINTAIN VOICE: Keep casual if casual, formal if formal, don't over-formalize
4. ADD MISSING ELEMENTS: Add context, audience, format, constraints, structure
5. BE SPECIFIC: Replace vague language with concrete, specific terms
6. ADD STRUCTURE: Break complex requests into numbered steps
7. REALISTIC LENGTH: Improved prompt should be 2-4x longer than original

YOUR IMPROVEMENTS MUST ADDRESS:
- CLARITY: Add specific action verbs, quantify elements (word counts, numbers), define all ambiguous terms
- CONTEXT: Add target audience (age, expertise, role), purpose/use case, relevant domain knowledge
- INSTRUCTIONS: Break into numbered steps (1, 2, 3...), add logical sequence, define sub-tasks clearly
- OUTPUT_FORMAT: Specify exact format (article, list, code, etc.), add length (word count), define structure (intro, body, conclusion), specify style/tone (formal, casual, technical)
- CONSTRAINTS: Add limitations ("max 500 words"), exclusions ("do not include..."), quality criteria ("cite sources"), edge case handling

CRITICAL: The improved prompt must be ACTIONABLE and SPECIFIC enough that any AI model would produce consistent, high-quality results.

Output JSON with:
- improved_prompt: The complete enhanced version (this should be 2-4x longer than the original)
- changes_made: Array of specific changes (limit to 5 most important)
- improvement_summary: One-sentence summary of key improvements
"""

improvement_template = ChatPromptTemplate.from_messages([
    ("system", improvement_system_prompt),
    ("human", """Original prompt:
{prompt}

Issues identified across all criteria:
{issues}

Current scores:
{scores}

Create an improved version that scores 9-10 on all criteria. Output only valid JSON.""")
])

# LCEL: Prompt -> LLM -> JSON Parser
improvement_chain = (
    improvement_template 
    | llm 
    | JsonOutputParser()
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def evaluate_and_improve(prompt: str):
    """
    Orchestrates the two chains:
    1. Runs evaluation chain to get scores and issues.
    2. Feeds evaluation results into improvement chain.
    """
    
    # Step 1: Evaluate
    try:
        eval_result = await evaluation_chain.ainvoke({"prompt": prompt})
    except Exception as e:
        logger.error(f"Evaluation chain failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze prompt logic.")

    # Step 2: Prepare data for improvement
    all_issues = []
    scores_dict = {}
    
    # Extract issues and scores safely
    for criterion, data in eval_result.items():
        if isinstance(data, dict):
            scores_dict[criterion] = data.get('score', 0)
            issues = data.get('issues', [])
            if issues:
                all_issues.extend(issues)
    
    # Step 3: Improve (using the evaluation context)
    try:
        improve_result = await improvement_chain.ainvoke({
            "prompt": prompt,
            "issues": "\n".join(all_issues) if all_issues else "No major issues, just optimize for perfection.",
            "scores": str(scores_dict)
        })
    except Exception as e:
        logger.error(f"Improvement chain failed: {e}")
        # Fallback if improvement fails: return original with empty changes
        improve_result = {
            "improved_prompt": prompt,
            "changes_made": ["Could not generate improvement due to server error"],
            "improvement_summary": "Original retained."
        }

    return eval_result, improve_result

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate_prompt(request: PromptInput):
    """
    Main endpoint: Evaluates a prompt and returns scores with improvements.
    Includes timeout handling and weighted scoring logic.
    """
    start_time = time.time()
    logger.info(f"Received evaluation request (len: {len(request.prompt)})")

    try:
        # Enforce 30s timeout on the logic
        eval_result, improve_result = await asyncio.wait_for(
            evaluate_and_improve(request.prompt), 
            timeout=30.0
        )

        # Calculate weighted overall score
        # Weights: Clarity (25%), Context (20%), Instructions (25%), Output (15%), Constraints (15%)
        weights = {
            "clarity": 0.25,
            "context": 0.20,
            "instructions": 0.25,
            "output_format": 0.15,
            "constraints": 0.15
        }
        
        overall_score = 0.0
        total_issues = 0
        
        # Safe calculation in case LLM hallucinates a key
        for criterion, weight in weights.items():
            if criterion in eval_result:
                score = eval_result[criterion].get("score", 0)
                overall_score += score * weight
                total_issues += len(eval_result[criterion].get("issues", []))
            else:
                # Fallback for missing keys
                overall_score += 5 * weight 

        overall_score = round(overall_score, 1)

        # Generate Explanation String
        changes = improve_result.get("changes_made", [])
        if isinstance(changes, list) and len(changes) > 0:
            top_changes = changes[:3]
            # Handle if changes are dicts or strings
            change_texts = [c.get("change", c) if isinstance(c, dict) else c for c in top_changes]
            changes_str = ", ".join(change_texts)
        else:
            changes_str = "enhanced structure and clarity"

        explanation = (
            f"Your prompt scored {overall_score}/10. "
            f"The improved version addresses {total_issues} issue{'s' if total_issues != 1 else ''}: "
            f"{changes_str}."
        )

        processing_time = round(time.time() - start_time, 2)
        
        logger.info(f"Success: Score {overall_score}, Time {processing_time}s")

        return EvaluationResponse(
            overall_score=overall_score,
            criteria=eval_result,
            improved_prompt=improve_result.get("improved_prompt", request.prompt),
            explanation=explanation,
            processing_time=processing_time
        )

    except asyncio.TimeoutError:
        logger.error("Request timed out after 30 seconds")
        raise HTTPException(
            status_code=504, 
            detail="Request timed out. The AI service took too long to respond."
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Internal server error during evaluation."
        )

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "PromptJudge AI",
        "llm_model": "gpt-4o-mini"
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to PromptJudge AI API",
        "docs": "/docs",
        "evaluate_endpoint": "POST /api/evaluate"
    }

# Entry point for local debugging
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)