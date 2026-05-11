from configs.base import DomainConfig

sales_config = DomainConfig(
    domain="sales",
    fields=["intent", "objections", "product_category", "engagement_level", "journey_stage"],
    cluster_on="objections",
    system_prompt="""You are an expert sales analyst specializing in behavioral pattern recognition in customer conversations. \
Your task is to extract structured insights from sales conversations with precision and consistency.
Always return valid JSON only. No explanations, no preamble, no markdown.""",
    classification_prompt="""Analyze the following sales conversation and extract five fields per conversation.

STRICT RULES:
- Write all text fields as direct, varied descriptions — no prefixes, no "Customer has...", no "The customer is..."
- Each intent and objection must be structurally different — vary your sentence openings
- product_category must be a short, specific phrase (2-4 words) describing the actual product or service discussed
- engagement_level must be exactly one of: hot | warm | cold
- journey_stage must be exactly one of: awareness | consideration | objection | negotiation | close | lost
- Return ONLY a valid JSON array, nothing else

FIELD DEFINITIONS:
- intent: what the customer is trying to accomplish or evaluate in this conversation
- objections: the core friction point or concern blocking progress, if any (use "no objection raised" if none)
- product_category: the specific product or service category being discussed
- engagement_level: how engaged and ready-to-buy the customer appears (hot=ready, warm=interested, cold=disengaged)
- journey_stage: where the customer is in the buying journey
  - awareness: still learning what the product is or does
  - consideration: evaluating options, comparing, asking questions
  - objection: a specific concern is actively blocking progress
  - negotiation: aligned on product, working out terms or price
  - close: committed or ready to commit
  - lost: disengaged, declined, or conversation ended negatively

SCHEMA:
[
  {
    "intent": string — one concise sentence, no prefix, varied structure,
    "objections": string — core friction point described directly, no prefix,
    "product_category": string — short specific phrase (2-4 words),
    "engagement_level": string — hot | warm | cold,
    "journey_stage": string — awareness | consideration | objection | negotiation | close | lost
  }
]

EXAMPLES:
[
  {
    "intent": "ready to invest immediately, just needs help narrowing down dividend options",
    "objections": "pricing too high relative to perceived value",
    "product_category": "Dividend Investment Fund",
    "engagement_level": "hot",
    "journey_stage": "negotiation"
  },
  {
    "intent": "comparison shopping across competitors before making a final decision",
    "objections": "unclear whether features justify the cost",
    "product_category": "SaaS Analytics Tool",
    "engagement_level": "warm",
    "journey_stage": "consideration"
  },
  {
    "intent": "a past bad experience with a competitor is blocking trust in similar products",
    "objections": "doubts about company reputation and long-term reliability",
    "product_category": "Skincare Product",
    "engagement_level": "cold",
    "journey_stage": "objection"
  },
  {
    "intent": "specific pain point around slow internet is driving urgency to switch providers",
    "objections": "worried about hidden fees and contract lock-in",
    "product_category": "Home Internet Service",
    "engagement_level": "hot",
    "journey_stage": "objection"
  }
]

VALIDATION: Before returning, verify your response is valid JSON with no trailing commas, no extra text, and all required fields present."""
)
