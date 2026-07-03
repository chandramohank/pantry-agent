# Intelligent Pantry API – Full Analysis

## Endpoint Classification Table

| # | Endpoint | Method | Tool Name | Category | Intent | Action Type | Confidence Handling |
|---|----------|--------|-----------|----------|--------|-------------|---------------------|
| 1 | `/api/ai/vision/extract-save` | POST | `extract_ingredients_from_image` | Vision AI | Scan image → extract + save ingredients | **Write** (creates pantry items) | Require approval if < 80% |
| 2 | `/api/ai/vision/ask-image` | POST | `ask_about_uploaded_image` | Vision AI | Q&A about food image | **Read** | Not applicable |
| 3 | `/api/pantry` | GET | `get_pantry_inventory` | Pantry Management | Retrieve all pantry items | **Read** | N/A |
| 4 | `/api/pantry` | POST | `add_pantry_item` | Pantry Management | Add single ingredient | **Write** | Duplicate check |
| 5 | `/api/ai/extract` | POST | `extract_pantry_items_from_text` | AI Extraction | Parse text → structured items (preview) | **Read** | Low confidence items flagged |
| 6 | `/api/ai/extract-and-save` | POST | `extract_and_save_pantry_items` | AI Extraction | Parse text → save to pantry | **Write** | Bulk threshold check |
| 7 | `/api/ai/diet-planner` | POST | `create_diet_plan` | Diet Planning | Generate meal plan | **Compute** | N/A |
| 8 | `/api/ai/cooking-copilot` | POST | `cooking_copilot` | Cooking Copilot | Answer cooking questions | **Compute** | N/A |
| 9 | `/api/ai/recipes` | GET | `recommend_recipes` | Recipe Intelligence | Recommend pantry-based recipes | **Read** | N/A |
| 10 | `/api/ai/waste-analysis` | GET | `analyze_food_waste` | Waste Reduction | Calculate spoilage risk per item | **Read** | N/A |
| 11 | `/api/ai/waste-reduction` | GET | `recommend_waste_reduction` | Waste Reduction | Recipes for expiring items | **Read** | N/A |
| 12 | `/api/dashboard/waste` | GET | `waste_dashboard` | Monitoring | Aggregate waste statistics | **Read** | N/A |
| 13 | `/api/dashboard/waste/top-risk` | GET | `top_risk_items` | Monitoring | Top N items by spoilage risk | **Read** | N/A |
| 14 | `/api/ai/sustainability-insights` | GET | `sustainability_insights` | Sustainability | Carbon/eco impact analysis | **Compute** | N/A |

---

## Tool Taxonomy

### Tier 1 – Vision AI (Highest Priority)
Always invoked first when an image is present.

| Tool | Trigger Keywords | Output |
|------|-----------------|--------|
| `extract_ingredients_from_image` | scan, photo, image, fridge, shelf, receipt | Structured ingredient list + confidence |
| `ask_about_uploaded_image` | question about image content | Natural language answer |

### Tier 2 – Pantry Management
Core CRUD operations on the inventory.

| Tool | Trigger Keywords | Output |
|------|-----------------|--------|
| `get_pantry_inventory` | what do I have, show pantry, list ingredients | Paginated item list |
| `add_pantry_item` | add, store, save (single item) | Created item |
| `extract_pantry_items_from_text` | I bought, shopping list (preview) | Parsed items (not saved) |
| `extract_and_save_pantry_items` | add from list, save all (confirmed) | Saved + skipped items |

### Tier 3 – Intelligence Layer
AI-powered analysis and recommendations.

| Tool | Trigger Keywords | Output |
|------|-----------------|--------|
| `recommend_recipes` | what can I cook, recipe ideas | Ranked recipe list |
| `create_diet_plan` | meal plan, diet, week of meals | Day-by-day meal structure |
| `cooking_copilot` | how do I, how long, substitute | Cooking guidance |

### Tier 4 – Waste & Sustainability
Environmental and waste management.

| Tool | Trigger Keywords | Output |
|------|-----------------|--------|
| `analyze_food_waste` | expiring, going bad, waste risk | Per-item risk scores |
| `recommend_waste_reduction` | use expiring items, reduce waste | Recipes for at-risk items |
| `waste_dashboard` | waste summary, pantry health | Aggregate counts |
| `top_risk_items` | urgent items, use today | Ordered risk list |
| `sustainability_insights` | eco, carbon, sustainability | Impact analysis |

---

## Domain → Tool Routing Matrix

| Domain | Primary Tools | Secondary Tools |
|--------|---------------|-----------------|
| Vision | extract_ingredients_from_image, ask_about_uploaded_image | add_pantry_item |
| Pantry | get_pantry_inventory, add_pantry_item | extract_pantry_items_from_text |
| Recipes | recommend_recipes | get_pantry_inventory |
| Diet | create_diet_plan | get_pantry_inventory |
| Cooking | cooking_copilot | get_pantry_inventory |
| Waste | analyze_food_waste, recommend_waste_reduction, waste_dashboard, top_risk_items | get_pantry_inventory |
| Sustainability | sustainability_insights | analyze_food_waste, waste_dashboard |
| General | All tools | — |

---

## Human-in-the-Loop Triggers

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Vision confidence | < 80% | Pause before saving; show items for review |
| Bulk add | ≥ 5 items | Require approval before write |
| Suspicious quantity | qty > 500 | Flag as validation error |
| API error on write | Any 4xx/5xx | Surface error; do not retry silently |
| Destructive operation | Any DELETE | Always require explicit confirmation |
