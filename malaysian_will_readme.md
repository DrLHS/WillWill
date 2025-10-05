# Malaysian Will Generation System - Complete Documentation

## Overview

This is a comprehensive **Retrieval Augmented Generation (RAG)** system for generating legally valid wills for non-Muslim Malaysians under the Wills Act 1959. The system combines structured legal knowledge with flexible LLM generation, using **JAN.ai** as the default provider but designed to work with any OpenAI-compatible API.

## System Architecture

The system follows a modular RAG architecture with five key components working together:

### 1. Knowledge Base Layer
The foundation contains extracted legal requirements from Malaysian will law, organized into structured documents. This includes requirements from the Wills Act 1959, standard clause templates from professional will writers like Rockwills and AmanahRaya, asset categorization rules, and validation requirements to ensure legal compliance.

### 2. Vector Store Layer
Using **ChromaDB** for persistent storage, the system creates semantic embeddings of all knowledge base documents using HuggingFace's sentence transformers. These embeddings run locally without requiring API calls and enable intelligent retrieval of relevant legal information based on semantic similarity rather than just keyword matching.

### 3. Retrieval Layer
The **LangChain RetrievalQA** chain connects the vector store to the language model. When you ask a question or need to generate a clause, it retrieves the four most relevant documents from the knowledge base and passes them as context to the LLM, ensuring responses are grounded in actual Malaysian law rather than hallucinated information.

### 4. Validation Layer
Built with **Pydantic** models, this layer provides automatic data validation with type checking and format validation. It performs critical legal checks like ensuring witnesses are not beneficiaries according to Section 9 of the Wills Act, verifying NRIC format compliance with Malaysian standards, checking age requirements, and validating that all mandatory will sections are present.

### 5. Generation Layer
The document generator assembles validated information into a properly structured will following the exact legal order required by Malaysian law, using legally precise language from actual Malaysian wills, including all mandatory clauses and sections, and generating proper attestation sections for execution.

## Why This Architecture?

### RAG vs Pure LLM Generation

You might wonder why we use RAG instead of just asking the LLM to generate a will. There are several critical reasons. **Legal precision** is paramount because LLMs can hallucinate legal requirements or use incorrect terminology that could invalidate a will. RAG grounds generation in verified legal sources. **Consistency** matters because Malaysian wills must follow specific formats and language. RAG ensures every generated clause matches established patterns. **Auditability** is important as you can trace every clause back to its source in the knowledge base, crucial for legal documents. **Updateability** means when laws change, you update the knowledge base rather than retraining the entire model.

### Modular Design Benefits

The modular architecture provides several advantages. You can **swap LLM providers** easily by changing just the configuration, moving from JAN.ai to OpenAI or Claude without touching the core logic. You can **update legal knowledge** by modifying the knowledge base documents without changing code. You can **extend functionality** by adding new asset types or clause templates to the knowledge base. Each component can be **tested independently**, making the system more reliable.

## Setup Instructions

### Prerequisites

Before starting, ensure you have Python 3.8 or higher installed, JAN.ai running locally on port 1337 (or another OpenAI-compatible LLM), and at least 4GB of free disk space for the vector store and models.

### Step-by-Step Setup

First, install JAN.ai by downloading it from jan.ai and installing it on your system. Launch JAN.ai and download a model like Mistral 7B Instruct or any other model you prefer. Start the local API server in JAN.ai settings, ensuring the API endpoint is set to http://localhost:1337.

Next, install Python dependencies. Open a terminal in your project directory and run the installation command to get all required packages including LangChain, ChromaDB, and Pydantic.

Then, initialize the system by running the Jupyter notebook cells in order from Part 1 through Part 4. The first run will download the embedding model, which takes a few minutes, and create the vector store from the knowledge base, which also takes time on the first run but is much faster on subsequent runs.

Finally, test the system by generating a sample will to verify everything works correctly.

### Configuration Options

You can customize the system through the WillGeneratorConfig class. For the LLM provider, set it to "jan.ai" for local JAN.ai, "openai" for OpenAI's API, or "anthropic" for Claude. The model name should match your JAN.ai model name or the model identifier from your provider. The API base URL is typically http://localhost:1337/v1 for JAN.ai or the provider's API endpoint for cloud services.

Temperature controls randomness, where lower values from zero to point three give more deterministic outputs, which is better for legal documents, while higher values from point seven to one point zero give more creative outputs, which you generally want to avoid for wills. The embedding model defaults to sentence-transformers/all-MiniLM-L6-v2, which is fast and accurate, but you can use other sentence transformer models for potentially better retrieval.

## Using the System

### Interactive Mode (Recommended for End Users)

For individual will creation with guided questions, run the questionnaire system. This walks users through each section, providing help when needed, validating inputs immediately, and catching common errors before generation.

### Programmatic Mode (For Integration)

When integrating with web applications or APIs, you would create data objects directly from form submissions, validate using Pydantic models, generate the will, and return it as text, PDF, or other formats.

### Batch Processing

For generating multiple wills from a database, you would load beneficiary and testator data from your source, create WillData objects for each person, generate all wills in a loop, and save with systematic filenames for organization.

## Customization Guide

### Adding New Asset Types

To support additional asset categories, first add the asset category to the knowledge base by creating a new document in the get_asset_categories method with the category name, legal requirements, and sample clauses. Then update the AssetInfo model by adding the new type to the validator and defining required specific details. Finally, add generation logic by creating a new clause generation method in the generator.

### Modifying Clause Templates

When you need to adjust the legal language used, locate the relevant section in the knowledge base documents, modify the template text while maintaining legal precision, regenerate the vector store to pick up changes, and test thoroughly to ensure the new language works correctly.

### Supporting Multiple Languages

To add support for Bahasa Malaysia or Chinese, create parallel knowledge bases for each language, add a language field to the configuration, load the appropriate knowledge base based on user preference, and generate clauses in the selected language.

### Adding Trust Provisions

For more sophisticated estate planning, extend the BeneficiaryInfo model to include trust parameters like vesting age, allowance structure, and trustee powers. Add trust clause templates to the knowledge base and implement trust clause generation in the will generator.

## Important Legal Considerations

### What This System Does

This system generates **draft wills** based on user input and Malaysian legal requirements. It validates inputs against Wills Act 1959 requirements, uses legally precise language from professional will writers, includes all mandatory sections and clauses, and provides signing instructions for proper execution.

### What This System Does NOT Do

It is crucial to understand the limitations. This system does **not** provide legal advice or recommendations about what should go in your will. It does not replace consultation with a qualified lawyer, especially for complex estates. It does not handle Muslim inheritance matters, which must follow Syariah law through a Wasiat. It cannot guarantee legal validity without professional review, and it does not provide estate planning strategy or tax optimization advice.

### Recommended Workflow

The best practice workflow involves several steps. Users should first generate their draft will using this system, gathering all necessary information about assets and beneficiaries. Then they should **review it carefully** with family members or trusted advisors to ensure all wishes are captured. The critical next step is to **consult a lawyer** who specializes in estate planning and probate in Malaysia for professional review. This typically costs between RM500 and RM1500 for straightforward estates but is essential for legal peace of mind.

After lawyer review and any necessary revisions, the will should be executed properly according to the signing instructions, with the testator and two witnesses present simultaneously, using original signatures in blue or black ink. Finally, the original will should be stored safely in a secure location like a bank safe deposit box or professional custody service, with the executor informed of the location but ideally not holding the original if they are also a beneficiary.

## Technical Deep Dive

### How RAG Retrieval Works

When you ask a question or request clause generation, the system follows a specific process. Your query is first converted to an embedding vector using the same sentence transformer model used for the knowledge base. The system then performs similarity search in ChromaDB, comparing your query vector against all knowledge base vectors and retrieving the four most semantically similar documents.

These retrieved documents are then formatted as context and inserted into a prompt template along with your original question. The LLM receives both the context and the question, generating a response grounded in the retrieved legal information. The sources are tracked and can be returned for transparency, allowing you to verify where information came from.

### Why Local Embeddings?

We use HuggingFace sentence transformers locally rather than calling an API for several important reasons. Privacy is paramount because will information is highly sensitive and should never leave your system unnecessarily. Cost efficiency matters as embedding hundreds of documents would be expensive with API pricing, while local embeddings are free after the initial model download. Speed is a factor since local inference is much faster than API calls, especially for batch operations. Finally, offline capability means the system works without internet connectivity once the model is downloaded.

### Validation Pipeline

The system performs validation at multiple stages to ensure quality. Input validation happens immediately when data is entered, checking NRIC format, age requirements, and required fields. Structural validation occurs before will generation, verifying all mandatory sections are present, checking the correct order of clauses, and ensuring proper legal terminology.

Legal validation checks critical compliance issues like witness-beneficiary conflicts under Section 9, executor appointment requirements, and guardian appointments for minors. Post-generation validation reviews the complete document to confirm all sections are properly formatted, signatures blocks are correct, and attestation language is legally compliant. Finally, the system generates signing instructions to ensure proper execution.

## Common Issues and Solutions

### JAN.ai Connection Issues

If the system cannot connect to JAN.ai, first verify that JAN.ai is running and the API server is started in settings. Check that the API endpoint is correct at http://localhost:1337/v1. Ensure no firewall is blocking port 1337, and try restarting JAN.ai if connection problems persist.

### Vector Store Errors

When encountering vector database issues, you can force recreation of the vector store by passing force_recreate equals True to the create_vector_store method. Delete the vector_store directory and regenerate if corrupted. Ensure you have write permissions to the project directory, and check that you have sufficient disk space available.

### Validation Failures

If beneficiary-witness conflicts occur, choose different witnesses who are not receiving anything in the will. For NRIC format errors, ensure the format is exactly twelve digits with optional dashes in the pattern YYMMDD-PB-###G. When missing required fields appear, check that all mandatory testator information is provided, ensure executor and witness details are complete, and verify that asset-specific details are included based on asset type.

### Generation Quality Issues

If clauses seem incorrect or inappropriate, check the knowledge base documents for accuracy, verify that the correct asset type was selected, and ensure beneficiary relationships are specified correctly. Try regenerating with different phrasing if results are unsatisfactory. Remember that the quality of output depends heavily on the quality and completeness of the knowledge base.

## Future Enhancements

Several improvements could be made to extend this system. Supporting **Wasiat generation** for Muslims would require adding Faraid calculation logic, Syariah law requirements, and integration with Islamic scholars for verification. Creating a **web interface** using Streamlit or Flask would provide a better user experience with form-based input, real-time validation feedback, and PDF generation capabilities.

Adding **PDF generation** with proper formatting would involve using ReportLab or similar libraries to create professional-looking documents with proper legal formatting and embedded signatures. Implementing **multi-language support** for Bahasa Malaysia and Chinese would make the system accessible to more Malaysians. Creating an **API service** with FastAPI or similar frameworks would enable integration with existing websites or applications through RESTful endpoints with authentication and usage tracking.

Finally, adding **electronic signature integration** with services like DocuSign could streamline the execution process, though you would need to verify legal validity of e-signatures for wills in Malaysia, which may require legislative changes.

## Production Deployment Checklist

Before deploying this system for real users, ensure you complete several critical steps. Add comprehensive **error handling** for all user inputs, network failures, and file system operations. Implement **logging** to track system usage, errors, and generated documents for audit purposes.

Add **user authentication** if deploying as a web service to protect sensitive information. Implement **data encryption** for stored will documents and personal information. Create comprehensive **backup systems** for the vector store and generated documents to prevent data loss.

Add **legal disclaimers** prominently displayed throughout the interface making it clear this generates drafts requiring lawyer review. Implement **version control** for knowledge base updates to track changes to legal requirements over time. Create a thorough **testing suite** with unit tests for all validation logic and integration tests for the complete workflow, along with legal review of sample outputs to ensure correctness.

Finally, establish **monitoring and alerting** systems to track system health and performance, alert on errors or unusual patterns, and monitor storage usage and API costs if applicable.

## Legal Disclaimer

**IMPORTANT**: This system generates draft wills for informational purposes only. It does **NOT** constitute legal advice. Wills are serious legal documents with significant consequences if done incorrectly. You should **ALWAYS** consult a qualified lawyer specializing in estate planning and probate in Malaysia before signing any will.

The developers and operators of this system accept no liability for any consequences arising from the use of generated documents. Malaysian law is complex and individual circumstances vary greatly. What works for one person may be legally inadequate or inappropriate for another.

Professional legal review typically costs between RM500 and RM1500 for straightforward estates but provides essential peace of mind and legal validity assurance. This relatively small investment can save your beneficiaries thousands of ringgit and months of legal complications after your death.

## Support and Contribution

This system is designed to be modular and extensible. If you improve it or add features, consider sharing your enhancements with the community. When reporting issues, include your configuration settings, the error message or unexpected behavior, steps to reproduce the problem, and your Python and package versions.

For legal questions about Malaysian wills, consult the Malaysian Bar Council, professional will writing services like Rockwills or AmanahRaya, or qualified estate planning lawyers. For technical questions about the system, review the code comments which explain the reasoning behind design decisions, check the knowledge base documents to understand the legal requirements, and experiment with the RAG system to see how retrieval affects generation quality.

## Conclusion

This Malaysian Will Generation System demonstrates how RAG architecture can be applied to legal document generation, combining the flexibility of LLMs with the precision of structured legal knowledge. By keeping legal requirements in a maintainable knowledge base, using local embeddings for privacy and efficiency, validating rigorously at every step, and generating with legally precise language, the system produces high-quality draft wills that can serve as an excellent starting point for professional legal review.

Remember that technology augments but does not replace human expertise, especially in legal matters. Use this system as a tool to organize your estate planning thoughts and create a solid draft, but always seek professional legal advice before executing any will.

Your family's future security is worth the investment in proper legal review.
