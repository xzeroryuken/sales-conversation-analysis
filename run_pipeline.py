from configs.sales import sales_config
from pipeline.core import ConversationPipeline
from pipeline.clustering import ClusteringPipeline
from pipeline.labeling import LabelingPipeline
from pipeline.llm import GroqClient
from pipeline.ingestion import HuggingFaceSource

EXTRACTION_OUTPUT = "classified_conversations.csv"
TAXONOMY_OUTPUT = "cluster_taxonomy.csv"

llm = GroqClient()

# Step 1: Extract structured fields per conversation
extraction = ConversationPipeline(
    config=sales_config,
    llm=llm,
    source=HuggingFaceSource("goendalf666/sales-conversations"),
    output_path=EXTRACTION_OUTPUT,
)
df = extraction.run()

# Step 2: Embed, reduce, and cluster
clustering = ClusteringPipeline(config=sales_config)
df = clustering.run(df)
df.to_csv(EXTRACTION_OUTPUT, index=False)

# Step 3: Label each cluster → taxonomy
labeling = LabelingPipeline(config=sales_config, llm=llm)
taxonomy = labeling.run(df)
taxonomy.to_csv(TAXONOMY_OUTPUT, index=False)

print("\nTaxonomy:")
print(taxonomy.to_string(index=False))
