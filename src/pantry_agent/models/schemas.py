"""Pydantic models for all Intelligent Pantry API request / response payloads."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class IngredientCategory(str, Enum):
    PRODUCE = "produce"
    DAIRY = "dairy"
    MEAT = "meat"
    SEAFOOD = "seafood"
    BAKERY = "bakery"
    FROZEN = "frozen"
    CANNED = "canned"
    DRY_GOODS = "dry_goods"
    CONDIMENTS = "condiments"
    BEVERAGES = "beverages"
    SNACKS = "snacks"
    OTHER = "other"


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class DietType(str, Enum):
    STANDARD = "standard"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    KETO = "keto"
    PALEO = "paleo"
    GLUTEN_FREE = "gluten_free"
    HIGH_PROTEIN = "high_protein"
    LOW_CARB = "low_carb"
    MEDITERRANEAN = "mediterranean"


# ── Shared primitives ─────────────────────────────────────────────────────────

class PantryItem(BaseModel):
    id: str | None = None
    name: str = Field(..., description="Ingredient name, e.g. 'whole milk'")
    quantity: float = Field(..., gt=0, description="Numeric amount")
    unit: str = Field(..., description="Unit of measure, e.g. 'litre', 'kg', 'pieces'")
    category: IngredientCategory = IngredientCategory.OTHER
    expiry_date: date | None = None
    location: str | None = Field(default=None, description="fridge | pantry | freezer")
    notes: str | None = None


class Recipe(BaseModel):
    id: str | None = None
    name: str
    description: str | None = None
    ingredients: list[dict[str, Any]] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    prep_time_minutes: int | None = None
    cook_time_minutes: int | None = None
    servings: int | None = None
    cuisine: str | None = None
    tags: list[str] = Field(default_factory=list)


# ── Vision ────────────────────────────────────────────────────────────────────

class VisionExtractRequest(BaseModel):
    """POST /api/ai/vision/extract-save"""
    image_data: str | None = Field(
        default=None, description="Base64-encoded image string (JPEG/PNG/WebP)"
    )
    image_url: str | None = Field(
        default=None, description="Presigned or public URL to the image"
    )
    auto_save: bool = Field(
        default=True, description="Automatically persist extracted items to pantry"
    )


class VisionExtractResponse(BaseModel):
    extracted_items: list[PantryItem]
    confidence: float = Field(..., ge=0.0, le=1.0)
    image_description: str | None = None
    warnings: list[str] = Field(default_factory=list)
    saved: bool = False


class VisionAskRequest(BaseModel):
    """POST /api/ai/vision/ask-image"""
    image_data: str | None = None
    image_url: str | None = None
    question: str = Field(..., description="Natural-language question about the image")


class VisionAskResponse(BaseModel):
    answer: str
    confidence: float | None = None
    detected_items: list[str] = Field(default_factory=list)


# ── Pantry management ─────────────────────────────────────────────────────────

class AddPantryItemRequest(BaseModel):
    """POST /api/pantry"""
    name: str
    quantity: float = Field(..., gt=0)
    unit: str
    category: IngredientCategory = IngredientCategory.OTHER
    expiry_date: date | None = None
    location: str | None = None
    notes: str | None = None


class PantryListResponse(BaseModel):
    items: list[PantryItem]
    total: int


# ── AI Extraction ─────────────────────────────────────────────────────────────

class ExtractTextRequest(BaseModel):
    """POST /api/ai/extract"""
    text: str = Field(..., description="Free-form text containing ingredient mentions")


class ExtractTextResponse(BaseModel):
    extracted_items: list[PantryItem]
    raw_text: str
    confidence: float | None = None


class ExtractAndSaveRequest(BaseModel):
    """POST /api/ai/extract-and-save"""
    text: str


class ExtractAndSaveResponse(BaseModel):
    extracted_items: list[PantryItem]
    saved_items: list[PantryItem]
    skipped_items: list[PantryItem] = Field(default_factory=list)
    message: str


# ── Diet planner ──────────────────────────────────────────────────────────────

class DietPlanRequest(BaseModel):
    """POST /api/ai/diet-planner"""
    diet_type: DietType = DietType.STANDARD
    days: int = Field(default=7, ge=1, le=30)
    calories_per_day: int | None = Field(default=None, ge=500, le=5000)
    allergies: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    use_pantry_items: bool = True


class MealPlan(BaseModel):
    day: int
    breakfast: Recipe | None = None
    lunch: Recipe | None = None
    dinner: Recipe | None = None
    snacks: list[Recipe] = Field(default_factory=list)


class DietPlanResponse(BaseModel):
    meal_plans: list[MealPlan]
    available_ingredients: list[str]
    missing_ingredients: list[str]
    substitutions: dict[str, str] = Field(default_factory=dict)
    shopping_list: list[PantryItem] = Field(default_factory=list)
    summary: str


# ── Cooking copilot ───────────────────────────────────────────────────────────

class CookingCopilotRequest(BaseModel):
    """POST /api/ai/cooking-copilot"""
    question: str = Field(..., description="Cooking question or instruction request")
    context: str | None = Field(
        default=None,
        description="Additional context: recipe name, ingredients on hand, etc.",
    )
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


class CookingCopilotResponse(BaseModel):
    answer: str
    tips: list[str] = Field(default_factory=list)
    substitutions: dict[str, str] = Field(default_factory=dict)
    related_recipes: list[str] = Field(default_factory=list)


# ── Recipes ───────────────────────────────────────────────────────────────────

class RecipeRecommendationResponse(BaseModel):
    """GET /api/ai/recipes"""
    recipes: list[Recipe]
    pantry_coverage_pct: float | None = None
    message: str | None = None


class RecipeSearchResult(BaseModel):
    """Single recipe result from hybrid search."""
    title: str
    url: str
    image: str | None = None
    total_time: int | None = None  # minutes
    calories: float | None = None
    protein: float | None = None
    hybrid_score: float = Field(..., ge=0.0, le=1.0, description="Blended BM25 + semantic score")


class RecipeSearchInput(BaseModel):
    """Input schema for recipe_search_tool."""
    query: str = Field(..., description="Natural language recipe search query (e.g., 'hearty beef stew for winter')")
    filters: dict | None = Field(
        default=None,
        description="Optional filters dict. Keys: max_time (int), min_protein (float), max_calories (float), "
                    "max_sodium (float), exclude_ingredients (list[str]), author (str), "
                    "max_fat (float), max_carbohydrate (float), max_cholesterol (float), "
                    "min_fiber (float), max_sugar (float)",
    )


class RecipeSearchResponse(BaseModel):
    """Response from recipe_search_tool."""
    recipes: list[RecipeSearchResult]
    total_found: int | None = None
    query: str
    execution_time_ms: float | None = None


# ── Waste analysis ────────────────────────────────────────────────────────────

class WasteItem(BaseModel):
    pantry_item: PantryItem
    waste_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    days_until_expiry: int | None = None
    expiry_prediction: date | None = None
    recommended_action: str | None = None


class WasteAnalysisResponse(BaseModel):
    """GET /api/ai/waste-analysis"""
    waste_items: list[WasteItem]
    total_items_analysed: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    estimated_waste_value: float | None = None


class WasteReductionResponse(BaseModel):
    """GET /api/ai/waste-reduction"""
    recipes: list[Recipe]
    priority_items: list[str]
    tips: list[str]


class WasteDashboardResponse(BaseModel):
    """GET /api/dashboard/waste"""
    total_items: int
    high_risk: int
    medium_risk: int
    low_risk: int
    no_risk: int
    waste_score_avg: float
    last_updated: str | None = None


class TopRiskItem(BaseModel):
    item: PantryItem
    risk_level: RiskLevel
    waste_score: float
    days_until_expiry: int | None = None


class TopRiskResponse(BaseModel):
    """GET /api/dashboard/waste/top-risk"""
    items: list[TopRiskItem]
    total: int


# ── Sustainability ────────────────────────────────────────────────────────────

class SustainabilityInsight(BaseModel):
    category: str
    insight: str
    recommendation: str
    impact_score: float | None = None


class SustainabilityResponse(BaseModel):
    """GET /api/ai/sustainability-insights"""
    insights: list[SustainabilityInsight]
    recommendations: list[str]
    actions: list[str]
    summary: str
    overall_score: float | None = None
    carbon_footprint_estimate: str | None = None


# ── Agent UI contract ────────────────────────────────────────────────────────

class ActionKind(str, Enum):
    SUBMIT_SELECTION = "submit_selection"
    OPEN_DETAILS = "open_details"
    REQUEST_REFRESH = "request_refresh"
    REQUEST_APPROVAL = "request_approval"
    CONTINUE = "continue"


class UIAction(BaseModel):
    action_id: str
    label: str
    kind: ActionKind
    target: str = "agent"
    payload: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False


class UILayout(BaseModel):
    variant: str | None = None
    density: str | None = None
    media_position: str | None = None
    group_by: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UIArtifact(BaseModel):
    artifact_id: str
    type: Literal[
        "selection_list",
        "card_collection",
        "table",
        "chart",
        "form",
        "approval_prompt",
    ]
    title: str
    description: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    actions: list[UIAction] = Field(default_factory=list)
    layout: UILayout = Field(default_factory=UILayout)
    accessibility: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class AgentResponseEnvelope(BaseModel):
    schema_version: str = "1.0"
    thread_id: str
    message: str
    artifacts: list[UIArtifact] = Field(default_factory=list)
    actions: list[UIAction] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    approval: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
