"""
Malaysian Will Generation System
QUICK START GUIDE & PRACTICAL EXAMPLES

This notebook provides ready-to-use examples for common scenarios.
Copy and modify these examples for your specific needs.
"""

# ============================================================================
# EXAMPLE 1: Using JAN.ai (Default - Free, Local, Private)
# ============================================================================

print("="*70)
print("EXAMPLE 1: JAN.ai Setup (Recommended for Privacy)")
print("="*70)

# Step 1: Make sure JAN.ai is running
# - Download from: https://jan.ai
# - Launch the app
# - Go to Settings > Advanced > Enable API Server
# - Download a model (e.g., Mistral 7B Instruct)
# - Note the API endpoint (usually http://localhost:1337/v1)

from pathlib import Path

# Configuration for JAN.ai
jan_config = WillGeneratorConfig(
    llm_provider="jan.ai",
    model_name="mistral-ins-7b-q4",  # Replace with your model name from JAN.ai
    api_base="http://localhost:1337/v1",
    api_key="not-needed",  # JAN.ai doesn't require API key
    temperature=0.1,  # Low temperature for consistent legal language
    max_tokens=2000
)

print(f"✅ JAN.ai configuration created")
print(f"   Model: {jan_config.model_name}")
print(f"   Endpoint: {jan_config.api_base}")
print(f"   Temperature: {jan_config.temperature} (low = more deterministic)")

# ============================================================================
# EXAMPLE 2: Using OpenAI API (If you prefer cloud-based)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: OpenAI Setup (Alternative)")
print("="*70)

# Note: Requires OPENAI_API_KEY environment variable
# Set it in your terminal: export OPENAI_API_KEY="sk-..."

openai_config = WillGeneratorConfig(
    llm_provider="openai",
    model_name="gpt-4",  # or "gpt-3.5-turbo" for lower cost
    api_base="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY", "your-key-here"),
    temperature=0.1,
    max_tokens=2000
)

print("✅ OpenAI configuration created")
print(f"   Model: {openai_config.model_name}")
print("   ⚠️  Remember: Using cloud API sends data to OpenAI servers")
print("   Consider privacy implications for sensitive will information")

# ============================================================================
# EXAMPLE 3: Simple Will - Everything to Spouse
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Simple Will (Everything to Spouse)")
print("="*70)

def create_simple_will_to_spouse():
    """
    Simplest possible will: Leave everything to spouse.
    
    This is common for young married couples without children
    or complex assets. Quick to create, easy to execute.
    """
    
    # Use the JAN.ai config (or openai_config if you prefer)
    config = jan_config
    rag = WillRAGSystem(config)
    rag.create_vector_store(kb_documents)
    rag.setup_retrieval_qa()
    generator = MalaysianWillGenerator(rag)
    
    # Create simple will data
    will_data = WillData(
        testator=TestatorInfo(
            full_name="Ahmad bin Ibrahim",
            nric="850303-14-2345",
            address="56 Jalan Raja Laut, 50350 Kuala Lumpur",
            religion="Non-Muslim",
            marital_status="Married"
        ),
        executors=[
            PersonInfo(
                full_name="Siti binti Ahmad",
                nric="870505-14-3456",
                address="56 Jalan Raja Laut, 50350 Kuala Lumpur",
                relationship="Wife"
            )
        ],
        witnesses=[
            PersonInfo(
                full_name="Lee Chong Wei",
                nric="750101-10-1111",
                address="12 Jalan Ampang, 50450 KL"
            ),
            PersonInfo(
                full_name="Muthu Krishnan",
                nric="800606-10-2222",
                address="34 Jalan Tun Razak, 50400 KL"
            )
        ],
        beneficiaries=[
            BeneficiaryInfo(
                full_name="Siti binti Ahmad",
                nric="870505-14-3456",
                address="56 Jalan Raja Laut, 50350 KL",
                relationship="Wife",
                distribution_type="residuary",
                distribution_value=None  # Gets everything
            )
        ],
        assets=[],  # No specific assets - using general residuary clause
        guardians=[],
        has_minor_children=False,
        special_instructions=None
    )
    
    # Generate and save
    will_text = generator.generate_will(will_data)
    will_path = generator.save_will(will_text, "simple_will_to_spouse.txt")
    
    print(f"✅ Simple will generated: {will_path}")
    print("\nKey features:")
    print("  - Everything goes to spouse via residuary clause")
    print("  - Spouse is also the executor")
    print("  - No specific asset listing needed")
    print("  - Perfect for young couples starting out")
    
    return will_text

# ============================================================================
# EXAMPLE 4: Will with Children - Equal Distribution
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Will with Children (Equal Shares)")
print("="*70)

def create_will_with_children():
    """
    Common scenario: Married with children, distribute equally.
    
    This example shows:
    - Multiple beneficiaries with equal shares
    - Guardian appointment for minor children
    - Handling spouse + children distribution
    """
    
    config = jan_config
    rag = WillRAGSystem(config)
    rag.create_vector_store(kb_documents)
    rag.setup_retrieval_qa()
    generator = MalaysianWillGenerator(rag)
    
    will_data = WillData(
        testator=TestatorInfo(
            full_name="Kumar Subramaniam",
            nric="750808-10-4567",
            address="23 Jalan Gasing, 46000 Petaling Jaya, Selangor",
            religion="Hindu",
            marital_status="Married"
        ),
        executors=[
            PersonInfo(
                full_name="Lakshmi Devi",
                nric="780909-10-5678",
                address="23 Jalan Gasing, 46000 PJ",
                relationship="Wife"
            ),
            PersonInfo(
                full_name="Rajan Subramaniam",
                nric="500505-10-9999",
                address="45 Jalan SS2, 47300 PJ",
                relationship="Brother"
            )
        ],
        witnesses=[
            PersonInfo(
                full_name="Tan Ah Kow",
                nric="720101-10-1234",
                address="12 Jalan University, 46200 PJ"
            ),
            PersonInfo(
                full_name="Wong Mei Ling",
                nric="800202-10-2345",
                address="34 Jalan 14/22, 46100 PJ"
            )
        ],
        beneficiaries=[
            BeneficiaryInfo(
                full_name="Lakshmi Devi",
                nric="780909-10-5678",
                address="23 Jalan Gasing, 46000 PJ",
                relationship="Wife",
                distribution_type="percentage",
                distribution_value="50%"
            ),
            BeneficiaryInfo(
                full_name="Arun Kumar",
                nric="100303-10-3333",
                address="23 Jalan Gasing, 46000 PJ",
                relationship="Son",
                distribution_type="percentage",
                distribution_value="25%"
            ),
            BeneficiaryInfo(
                full_name="Priya Kumar",
                nric="120505-10-4444",
                address="23 Jalan Gasing, 46000 PJ",
                relationship="Daughter",
                distribution_type="percentage",
                distribution_value="25%"
            )
        ],
        assets=[],  # Using percentage distribution of residuary estate
        guardians=[
            PersonInfo(
                full_name="Rajan Subramaniam",
                nric="500505-10-9999",
                address="45 Jalan SS2, 47300 PJ",
                relationship="Uncle"
            )
        ],
        has_minor_children=True,
        special_instructions="I wish my children to be raised in the Hindu faith."
    )
    
    will_text = generator.generate_will(will_data)
    will_path = generator.save_will(will_text, "will_with_children_equal_shares.txt")
    
    print(f"✅ Family will generated: {will_path}")
    print("\nKey features:")
    print("  - 50% to spouse, 25% to each child")
    print("  - Guardian appointed for minor children")
    print("  - Brother as backup executor")
    print("  - Special instructions for children's upbringing")
    
    return will_text

# ============================================================================
# EXAMPLE 5: Will with Specific Assets
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 5: Will with Specific Asset Distributions")
print("="*70)

def create_will_with_specific_assets():
    """
    Detailed will with specific asset allocations.
    
    Shows how to:
    - Distribute specific properties to specific people
    - Handle different asset types (property, car, bank accounts)
    - Use both specific bequests and residuary clause
    """
    
    config = jan_config
    rag = WillRAGSystem(config)
    rag.create_vector_store(kb_documents)
    rag.setup_retrieval_qa()
    generator = MalaysianWillGenerator(rag)
    
    will_data = WillData(
        testator=TestatorInfo(
            full_name="Lim Keng Soon",
            nric="650707-10-6789",
            address="78 Jalan Sultan Ismail, 50250 Kuala Lumpur",
            religion="Christian",
            marital_status="Married"
        ),
        executors=[
            PersonInfo(
                full_name="Lim Wei Jie",
                nric="900808-10-7890",
                address="78 Jalan Sultan Ismail, 50250 KL",
                relationship="Son"
            )
        ],
        witnesses=[
            PersonInfo(
                full_name="David Tan",
                nric="700909-10-8888",
                address="90 Jalan Bukit Bintang, 50200 KL"
            ),
            PersonInfo(
                full_name="Sarah Lee",
                nric="751010-10-9999",
                address="12 Jalan Imbi, 55100 KL"
            )
        ],
        beneficiaries=[
            BeneficiaryInfo(
                full_name="Mary Lim",
                nric="670606-10-1111",
                address="78 Jalan Sultan Ismail, 50250 KL",
                relationship="Wife",
                distribution_type="specific",
                distribution_value="House and bank accounts"
            ),
            BeneficiaryInfo(
                full_name="Lim Wei Jie",
                nric="900808-10-7890",
                address="78 Jalan Sultan Ismail, 50250 KL",
                relationship="Son",
                distribution_type="specific",
                distribution_value="Business and car"
            ),
            BeneficiaryInfo(
                full_name="Lim Mei Ling",
                nric="920909-10-2222",
                address="45 Jalan Ampang, 50450 KL",
                relationship="Daughter",
                distribution_type="residuary",
                distribution_value="Remainder"
            )
        ],
        assets=[
            AssetInfo(
                asset_type="real_property",
                description="Family home",
                value=1200000.0,
                specific_details={
                    'address': "78 Jalan Sultan Ismail, 50250 Kuala Lumpur",
                    'title_number': "HS(D) 98765 PT 43210",
                    'ownership_type': "sole"
                },
                beneficiary_nric="670606-10-1111"  # Wife
            ),
            AssetInfo(
                asset_type="bank_account",
                description="All Maybank accounts",
                value=300000.0,
                specific_details={
                    'bank_name': "Maybank Berhad",
                    'account_type': "All accounts"
                },
                beneficiary_nric="670606-10-1111"  # Wife
            ),
            AssetInfo(
                asset_type="business",
                description="Family trading business",
                value=800000.0,
                specific_details={
                    'company_name': "Lim Trading Sdn Bhd",
                    'registration_number': "123456-X",
                    'ownership_percentage': "100"
                },
                beneficiary_nric="900808-10-7890"  # Son
            ),
            AssetInfo(
                asset_type="vehicle",
                description="Family car",
                value=150000.0,
                specific_details={
                    'registration_number': "WXY 9876",
                    'make_model': "Mercedes E-Class 2022"
                },
                beneficiary_nric="900808-10-7890"  # Son
            )
        ],
        guardians=[],
        has_minor_children=False,
        special_instructions="I request burial at Nirvana Memorial Park."
    )
    
    will_text = generator.generate_will(will_data)
    will_path = generator.save_will(will_text, "will_with_specific_assets.txt")
    
    print(f"✅ Detailed asset will generated: {will_path}")
    print("\nKey features:")
    print("  - House and bank accounts to wife")
    print("  - Business and car to son")
    print("  - Everything else to daughter (residuary)")
    print("  - Specific descriptions for each asset")
    print("  - Clear succession plan for family business")
    
    return will_text

# ============================================================================
# EXAMPLE 6: Using RAG for Interactive Help
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 6: Using RAG System for Legal Questions")
print("="*70)

def demonstrate_rag_queries():
    """
    Show how to use the RAG system to answer legal questions.
    
    This is useful for:
    - Building a chatbot that explains Malaysian will law
    - Providing contextual help in your UI
    - Validating whether something is legally allowed
    """
    
    config = jan_config
    rag = WillRAGSystem(config)
    rag.create_vector_store(kb_documents)
    rag.setup_retrieval_qa()
    
    print("\n🤖 RAG System ready for questions!")
    print("\nAsking common legal questions...\n")
    
    # Question 1
    q1 = "Can my wife be both a beneficiary and a witness?"
    print(f"Q: {q1}")
    a1 = rag.query(q1)
    print(f"A: {a1[:200]}...\n")
    
    # Question 2
    q2 = "What happens if I get married after making this will?"
    print(f"Q: {q2}")
    a2 = rag.query(q2)
    print(f"A: {a2[:200]}...\n")
    
    # Question 3
    q3 = "How do I leave my EPF money to my children?"
    print(f"Q: {q3}")
    a3 = rag.query(q3)
    print(f"A: {a3[:200]}...\n")
    
    print("✅ RAG system can answer legal questions based on knowledge base")
    print("   Use this to build interactive help systems")
    print("   Or to validate user inputs against legal requirements")

# ============================================================================
# EXAMPLE 7: Batch Processing Multiple Wills
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 7: Batch Processing (Generate Multiple Wills)")
print("="*70)

def batch_process_wills():
    """
    Generate multiple wills from a list of people.
    
    Useful for:
    - Will writing services processing multiple clients
    - Corporate estate planning programs
    - Testing and validation
    """
    
    config = jan_config
    rag = WillRAGSystem(config)
    rag.create_vector_store(kb_documents)
    rag.setup_retrieval_qa()
    generator = MalaysianWillGenerator(rag)
    
    # Sample client data (would come from database in production)
    clients = [
        {
            "name": "Client A",
            "nric": "800101-10-1234",
            "spouse_nric": "820202-10-2345"
        },
        {
            "name": "Client B",
            "nric": "750303-14-3456",
            "spouse_nric": "770404-14-4567"
        }
    ]
    
    generated_wills = []
    
    for client in clients:
        # Create will data (simplified - would be more detailed in production)
        will_data = WillData(
            testator=TestatorInfo(
                full_name=client["name"],
                nric=client["nric"],
                address="Sample Address",
                religion="Non-Muslim",
                marital_status="Married"
            ),
            executors=[
                PersonInfo(
                    full_name=f"{client['name']} Spouse",
                    nric=client["spouse_nric"],
                    address="Sample Address",
                    relationship="Spouse"
                )
            ],
            witnesses=[
                PersonInfo(
                    full_name="Witness 1",
                    nric="700101-10-9999",
                    address="Witness Address"
                ),
                PersonInfo(
                    full_name="Witness 2",
                    nric="750202-10-8888",
                    address="Witness Address"
                )
            ],
            beneficiaries=[
                BeneficiaryInfo(
                    full_name=f"{client['name']} Spouse",
                    nric=client["spouse_nric"],
                    address="Sample Address",
                    relationship="Spouse",
                    distribution_type="residuary"
                )
            ],
            assets=[],
            guardians=[],
            has_minor_children=False
        )
        
        # Generate
        will_text = generator.generate_will(will_data)
        filename = f"batch_will_{client['name'].replace(' ', '_')}.txt"
        will_path = generator.save_will(will_text, filename)
        
        generated_wills.append({
            "client": client["name"],
            "path": will_path
        })
    
    print(f"✅ Batch processed {len(generated_wills)} wills:")
    for result in generated_wills:
        print(f"   - {result['client']}: {result['path']}")

# ============================================================================
# EXAMPLE 8: Validation-Only Mode
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 8: Validation-Only (Check without Generating)")
print("="*70)

def validate_will_data_only():
    """
    Validate will information without generating document.
    
    Useful for:
    - Real-time form validation in web UIs
    - Pre-flight checks before generation
    - Catching errors early in the process
    """
    
    # Create test data with intentional error
    will_data = WillData(
        testator=TestatorInfo(
            full_name="Test Person",
            nric="800101-10-1234",
            address="Test Address",
            religion="Non-Muslim",
            marital_status="Married"
        ),
        executors=[
            PersonInfo(
                full_name="Executor",
                nric="820202-10-2345",
                address="Executor Address",
                relationship="Spouse"
            )
        ],
        # ERROR: Using executor as witness!
        witnesses=[
            PersonInfo(
                full_name="Executor",  # Same person as executor
                nric="820202-10-2345",  # Also a beneficiary!
                address="Executor Address"
            ),
            PersonInfo(
                full_name="Witness 2",
                nric="700101-10-9999",
                address="Witness Address"
            )
        ],
        beneficiaries=[
            BeneficiaryInfo(
                full_name="Executor",
                nric="820202-10-2345",  # Will trigger validation error
                address="Executor Address",
                relationship="Spouse",
                distribution_type="residuary"
            )
        ],
        assets=[],
        guardians=[],
        has_minor_children=False
    )
    
    # Validate
    valid, errors = will_data.validate_complete()
    
    if valid:
        print("✅ All validation checks passed!")
    else:
        print("❌ Validation errors found:")
        for error in errors:
            print(f"   - {error}")
    
    print("\n💡 Use validation before showing 'Generate Will' button")
    print("   Guide users to fix errors before allowing generation")

# ============================================================================
# EXAMPLE 9: Web API Integration Pattern
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 9: Web API Integration (FastAPI Pattern)")
print("="*70)

print("""
# Example FastAPI endpoint for will generation:

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Initialize RAG system once at startup
@app.on_event("startup")
async def startup_event():
    global rag_system, will_generator
    config = WillGeneratorConfig(llm_provider="jan.ai")
    rag_system = WillRAGSystem(config)
    rag_system.create_vector_store(kb_documents)
    rag_system.setup_retrieval_qa()
    will_generator = MalaysianWillGenerator(rag_system)

# Request model
class WillRequest(BaseModel):
    testator: dict
    executors: list
    witnesses: list
    beneficiaries: list
    assets: list = []
    guardians: list = []
    has_minor_children: bool = False
    special_instructions: str = None

# Generate will endpoint
@app.post("/generate-will")
async def generate_will(request: WillRequest):
    try:
        # Convert request to WillData
        will_data = WillData(
            testator=TestatorInfo(**request.testator),
            executors=[PersonInfo(**e) for e in request.executors],
            witnesses=[PersonInfo(**w) for w in request.witnesses],
            beneficiaries=[BeneficiaryInfo(**b) for b in request.beneficiaries],
            assets=[AssetInfo(**a) for a in request.assets],
            guardians=[PersonInfo(**g) for g in request.guardians],
            has_minor_children=request.has_minor_children,
            special_instructions=request.special_instructions
        )
        
        # Validate
        valid, errors = will_data.validate_complete()
        if not valid:
            raise HTTPException(status_code=400, detail=errors)
        
        # Generate
        will_text = will_generator.generate_will(will_data)
        
        return {
            "status": "success",
            "will_text": will_text,
            "warnings": errors if errors else []
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload
""")

print("\n✅ Example API pattern provided")
print("   Adapt this for your web framework (Flask, Django, etc.)")

# ============================================================================
# RUN EXAMPLES
# ============================================================================

print("\n" + "="*70)
print("RUNNING QUICK START EXAMPLES")
print("="*70)
print("\nNote: In production, you would call these functions individually.")
print("For this demo, we'll just show the configuration patterns.\n")

# Show that system is ready
print("✅ All example patterns documented and ready to use!")
print("\nTo use in your code:")
print("  1. Choose your LLM provider (JAN.ai or OpenAI)")
print("  2. Copy the relevant example above")
print("  3. Modify the data for your specific use case")
print("  4. Run the generation function")
print("\nFor interactive use:")
print("  >>> run_complete_example()  # From Part 4")
print("\nFor programmatic use:")
print("  >>> create_simple_will_to_spouse()")
print("  >>> create_will_with_children()")
print("  >>> create_will_with_specific_assets()")
