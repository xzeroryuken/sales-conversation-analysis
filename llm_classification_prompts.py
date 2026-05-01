system_prompt = """You are an expert sales analyst specializing in analyzing customer conversations and extracting behavioral insights."""

classification_prompt = """Classify the following sales conversation based on the intent of the customer, objections raised, product category, and engagement level. 
            The classification should be based on the customer's behavior and intent during the sales conversation. 
            
            Follow these rules strictly:
            - intent: one concise sentence describing the customer's behavior and motivation, no prefix
            - objections: one concise sentence describing the core objection directly, no prefix or preamble
            - product_category: two words only
            - engagement_level: one word only (hot | warm | cold)
            
            Schema:
            - intent: string (one sentence, no prefix)
            - objections: string (one sentence, no prefix)
            - product_category: string (two words)
            - engagement_level: string (hot | warm | cold)
            
            Only return the classifications in json format, no preamble. Here are some examples:
            
            [
                {
                    "intent": "highly engaged and ready to invest, seeking specific dividend opportunities",
                    "objections": "pricing too high relative to perceived value",
                    "product_category": "Investment Product",
                    "engagement_level": "hot"
                },
                {
                    "intent": "exploring options but needs more information before committing",
                    "objections": "unclear product features and functionality",
                    "product_category": "Technology Product",
                    "engagement_level": "warm"
                },
                {
                    "intent": "hesitant after negative experience with a competitor product",
                    "objections": "doubts about company reputation and reliability",
                    "product_category": "Beauty Product",
                    "engagement_level": "cold"
                }
            ]"""