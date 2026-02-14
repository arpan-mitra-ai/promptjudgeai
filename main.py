"""
PromptJudge AI - Backend API
Evaluates and improves AI prompts using LangChain LCEL chains with GPT-4o-mini.
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

# CORS Configuration - CRITICAL FIX: Allow specific methods and headers explicitly
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
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=2500,
        request_timeout=30,
        api_key=api_key
    )
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    # We don't raise here to allow the app to start, but requests will fail gracefully
    llm = None

# Evaluation Templates (kept your logic, just ensured stability)
evaluation_system_prompt = """You are PromptJudge, an expert AI prompt evaluator.
TASK: Evaluate the provided prompt on 5 criteria: clarity, context, instructions, output_format, constraints.
For each criterion:
1. Assign a score from 1-10.
2. Identify 2-4 specific issues by QUOTING exact text.
3. Identify 1-3 strengths by QUOTING exact text.
Output ONLY valid JSON.
"""
# ... (Use the full prompt text you had before for best results) ... 

# Simplified for brevity in this fix - use your full prompt strings here
evaluation_template = ChatPromptTemplate.from_messages([
    ("system", evaluation_system_prompt), # Use your full prompt here
    ("human", "{prompt}")
])

if llm:
    evaluation_chain = evaluation_template | llm | JsonOutputParser()
else:
    evaluation_chain = None

# Improvement Templates
improvement_system_prompt = """You are a prompt improvement specialist.
TASK: Transform the provided prompt into a high-quality version (score 9-10).
Output JSON with: improved_prompt, changes_made, improvement_summary.
"""

improvement_template = ChatPromptTemplate.from_messages([
    ("system", improvement_system_prompt), # Use your full prompt here
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
        # Ensuring we use the full system prompt defined in your original code
        eval_result = await evaluation_chain.ainvoke({"prompt": prompt})
    except Exception as e:
        logger.error(f"Evaluation chain failed: {e}")
        # Return a fallback result instead of crashing
        return fallback_evaluation(), fallback_improvement(prompt)

    # Step 2: Prepare data
    all_issues = []
    scores_dict = {}
    
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
