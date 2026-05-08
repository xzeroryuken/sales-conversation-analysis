from configs.sales import sales_config
from pipeline.core import ConversationPipeline
from pipeline.llm import GroqClient
from pipeline.ingestion import HuggingFaceSource

pipeline = ConversationPipeline(
    config=sales_config,
    llm=GroqClient(),
    source=HuggingFaceSource("goendalf666/sales-conversations"),
    output_path="classified_conversations.csv",
)

pipeline.run()
