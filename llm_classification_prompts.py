system_prompt = """You are an expert sales analyst specializing in analyzing customer conversations and extracting behavioral insights."""

classification_prompt = """Classify the following sales conversation based on the intent of the customer, objections raised, product category, and engagement level. 
            The classification should be based on the customer's behavior and intent during the sales conversation. 
            Make sure to classify the intent in one sentence and be concise, the objections raised in one sentence, the product category in two words, and the engagement level in one word. 
            Schema:
            - intent: string (one sentence)
            - objections: string (one sentence)  
            - product_category: string (two words)
            - engagement_level: string (hot | warm | cold)
            Only return the classifications in json format. Here are some examples of classifications:
            
            [
                {
                    "intent": "Customer is highly engaged and interested in dividend investing",
                    "objections": "Customer has raised objections about pricing",
                    "product_category": "Investment Product",
                    "engagement_level": "hot"
                },
                {
                    "intent": "Customer is asking questions about the product but is not fully committed",
                    "objections": "Customer has raised objections about the product's features",
                    "product_category": "Technology Product",
                    "engagement_level": "warm"
                },
                {
                    "intent": "Customer is hesitant to trust products after a bad experience with a competitor",
                    "objections": "Customer has raised objections about the company's reputation",
                    "product_category": "Beauty Product",
                    "engagement_level": "cold"
                }
            ]"""