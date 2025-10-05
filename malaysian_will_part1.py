"""
Malaysian Will Generation System with RAG Architecture
Part 1: Setup, Knowledge Base Creation, and Document Processing

This notebook builds a complete will generation system for non-Muslim Malaysians
using RAG (Retrieval Augmented Generation) with JAN.ai as the LLM provider.

Author: Claude
Date: 2025
"""

# ============================================================================
# CELL 1: Install Required Dependencies
# ============================================================================

# First, let's install all the packages we'll need
# LangChain provides the RAG framework
# Chroma is our vector database for storing embeddings
# OpenAI package works with JAN.ai (OpenAI-compatible API)

!pip install langchain langchain-community langchain-openai
!pip install chromadb
!pip install python-dotenv
!pip install pandas numpy
!pip install pydantic

print("✅ All dependencies installed successfully!")

# ============================================================================
# CELL 2: Import Libraries and Configure Environment
# ============================================================================

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

# LangChain core components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Data handling
import pandas as pd
import numpy as np

# Pydantic for data validation
from pydantic import BaseModel, Field, validator

print("✅ Libraries imported successfully!")

# ============================================================================
# CELL 3: Configuration Class
# ============================================================================

class WillGeneratorConfig:
    """
    Configuration manager for the will generation system.
    
    This centralizes all settings and makes it easy to switch between
    different LLM providers (JAN.ai, OpenAI, etc.) without changing code.
    """
    
    def __init__(
        self,
        llm_provider: str = "jan.ai",
        model_name: str = "mistral-ins-7b-q4",
        api_base: str = "http://localhost:1337/v1",
        api_key: str = "not-needed",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize configuration.
        
        Parameters:
        -----------
        llm_provider : str
            The LLM provider name (jan.ai, openai, etc.)
        model_name : str
            Model identifier - for JAN.ai, use the model name from your setup
        api_base : str
            API endpoint - JAN.ai runs locally on http://localhost:1337/v1
        api_key : str
            API key - JAN.ai doesn't require one, but field is needed
        temperature : float
            Lower values (0.0-0.3) make output more deterministic - important for legal docs
        max_tokens : int
            Maximum response length
        embedding_model : str
            HuggingFace model for creating embeddings (free, runs locally)
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.embedding_model = embedding_model
        
        # Storage paths
        self.knowledge_base_path = Path("./knowledge_base")
        self.vector_store_path = Path("./vector_store")
        self.output_path = Path("./generated_wills")
        
        # Create directories if they don't exist
        self.knowledge_base_path.mkdir(exist_ok=True)
        self.vector_store_path.mkdir(exist_ok=True)
        self.output_path.mkdir(exist_ok=True)
    
    def get_llm(self):
        """
        Returns configured LLM instance.
        
        This method creates the language model connection. Because we're using
        ChatOpenAI with JAN.ai's OpenAI-compatible endpoint, switching to
        actual OpenAI later just requires changing the api_base and api_key.
        """
        return ChatOpenAI(
            model_name=self.model_name,
            openai_api_base=self.api_base,
            openai_api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def get_embeddings(self):
        """
        Returns embedding model for RAG.
        
        We use HuggingFace embeddings which run locally and are free.
        This converts text into numerical vectors that capture semantic meaning,
        allowing us to find relevant legal clauses and requirements.
        """
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'}
        )

# Initialize configuration
config = WillGeneratorConfig()
print(f"✅ Configuration initialized for {config.llm_provider}")
print(f"   Model: {config.model_name}")
print(f"   API Base: {config.api_base}")

# ============================================================================
# CELL 4: Malaysian Will Knowledge Base
# ============================================================================

class MalaysianWillKnowledgeBase:
    """
    Knowledge base containing Malaysian will requirements, legal clauses,
    and validation rules extracted from our research.
    
    This class organizes all the legal knowledge we gathered about Malaysian
    wills into structured documents that can be embedded and retrieved by RAG.
    """
    
    @staticmethod
    def get_legal_requirements() -> List[Dict[str, str]]:
        """
        Legal requirements from Wills Act 1959 for non-Muslims.
        
        Each requirement is stored as a separate document with metadata.
        This allows the RAG system to retrieve specific requirements
        when validating user inputs or generating will clauses.
        """
        return [
            {
                "category": "age_requirements",
                "content": """
                AGE REQUIREMENTS (Wills Act 1959, Section 3-4):
                - Testator must be minimum 18 years old (21 years in Sabah)
                - No will made by any person under the age of majority shall be valid
                - Exception: privileged wills for military personnel and mariners under Section 26
                - The testator must be of sound mind at the time of making the will
                - Sound mind means understanding: (1) making a will, (2) nature and effect of testamentary act,
                  (3) extent of property, (4) claims of potential beneficiaries, (5) free from insane delusions
                - Mental capacity challenges can be prevented by obtaining medical certificate if capacity may be questioned
                """
            },
            {
                "category": "witness_requirements",
                "content": """
                WITNESS REQUIREMENTS (Wills Act 1959, Section 5, 9, 10, 11):
                - Minimum 2 witnesses required
                - Both witnesses must be present at SAME TIME during execution
                - Both must subscribe (sign) in testator's presence
                - Witnesses should preferably be 21+ years (18 minimum)
                - Witnesses must be of sound mind
                
                CRITICAL DISQUALIFICATIONS (Section 9):
                - Beneficiaries CANNOT witness - violation makes their gift "utterly null and void"
                - Spouses of beneficiaries CANNOT witness - same consequence
                - ANY beneficial devise, legacy, estate, interest, gift becomes void if witness is beneficiary
                
                ALLOWED (Section 11):
                - Executors CAN witness if they are not also beneficiaries
                
                ALLOWED (Section 10):
                - Creditors CAN witness
                
                WITNESS DETAILS REQUIRED:
                - Full legal name
                - NRIC number (12-digit Malaysian format)
                - Complete address with postcode and state
                - Signature
                
                NO prescribed attestation form required by law, but witnesses must document:
                "Signed by the testator in our presence and by us in the testator's presence"
                """
            },
            {
                "category": "execution_requirements",
                "content": """
                EXECUTION REQUIREMENTS (Wills Act 1959, Section 5):
                - Will MUST be in writing (typed or handwritten)
                - Must be signed by testator at foot or end (some flexibility allowed)
                - Signature made or acknowledged in presence of 2+ witnesses present at same time
                - Witnesses must subscribe in testator's presence
                - Publication (declaring it to be a will) NOT necessary (Section 7)
                - NO stamping required for validity
                - Original will generally required for probate (copies insufficient unless original proven lost)
                
                SIGNATURE PLACEMENT:
                - At foot or end of will
                - Flexibility allowed if "apparent on face of will that testator intended to give effect by such signature"
                - Best practice: sign at very end after all clauses
                """
            },
            {
                "category": "revocation_rules",
                "content": """
                REVOCATION MECHANISMS (Wills Act 1959, Section 12-16):
                
                AUTOMATIC REVOCATION:
                - Marriage or remarriage automatically revokes will (Section 12)
                - Exception: will made "in contemplation of marriage" naming intended spouse
                - Conversion to Islam (estate then follows Islamic law)
                - Writing new will with revocation clause
                
                VOLUNTARY REVOCATION:
                - Written declaration
                - Codicil (amendment document)
                - Intentional destruction with intent to revoke
                
                IMPORTANT NOTES:
                - Changed circumstances alone do NOT revoke will (Section 13)
                - Divorce does NOT automatically revoke will - must write new will
                - Revoked will only revived by re-execution or codicil showing revival intention (Section 16)
                
                STANDARD REVOCATION CLAUSE:
                "I hereby revoke all former wills and testamentary dispositions heretofore made by me"
                """
            },
            {
                "category": "mandatory_will_sections",
                "content": """
                MANDATORY WILL SECTIONS (Sequential Order):
                
                1. OPENING DECLARATION
                   - Testator's full legal name (as per NRIC)
                   - NRIC number (12-digit format: YYMMDD-PB-###G)
                   - Complete residential address
                   - Declaration of sound mind
                   
                2. REVOCATION CLAUSE
                   - Explicit revocation of all previous wills
                   - Standard language: "I DO HEREBY REVOKE all former testamentary dispositions,
                     wills or codicils herebefore made by me AND DECLARE this to be my LAST WILL AND TESTAMENT"
                
                3. EXECUTOR APPOINTMENT
                   - Name 1-4 executors with full details
                   - NRIC numbers and addresses
                   - Relationship to testator
                   - Alternate executors specified
                   
                4. DEBTS AND EXPENSES CLAUSE
                   - Direction to pay "just debts, funeral and testamentary expenses and estate duty (if any)"
                   - Must be paid BEFORE distribution
                
                5. DISTRIBUTION CLAUSES
                   - Specific bequests (individual gifts)
                   - Residuary estate clause (catch-all for remaining property)
                   - Use terminology: "I give, devise and bequeath"
                   
                6. GUARDIAN APPOINTMENT (if children under 18)
                   - Guardian name, NRIC, address
                   - Custody and care provisions
                   - Age limit: until children attain 18 years
                
                7. TESTIMONIUM CLAUSE
                   - Formal closing statement
                   - Date of execution
                   
                8. ATTESTATION SECTION
                   - Testator signature
                   - Two witness signatures
                   - All signatures with full details
                
                9. SIGNATURE BLOCKS
                   - Space for testator signature and date
                   - Space for two witnesses with names, NRIC, addresses, signatures
                """
            }
        ]
    
    @staticmethod
    def get_asset_categories() -> List[Dict[str, str]]:
        """
        Asset categories and handling requirements for Malaysian wills.
        
        Different asset types have different legal considerations in Malaysia.
        This knowledge helps the system generate appropriate clauses and warnings.
        """
        return [
            {
                "category": "real_property",
                "content": """
                REAL PROPERTY (Land and Buildings):
                
                REQUIRED INFORMATION:
                - Property type (residential/commercial/land)
                - Full address with postcode and state
                - Land title number (formats: HS(D) for strata, GM for master title, PN for provisional)
                - Ownership type (sole/joint tenancy/tenancy in common)
                - Ownership percentage if co-owned
                
                STANDARD CLAUSE:
                "I give, devise and bequeath my property known as [ADDRESS] held under Title No. [TITLE NUMBER]
                to my [RELATIONSHIP], [NAME] (NRIC No. [NUMBER]), absolutely."
                
                IMPORTANT CONSIDERATIONS:
                - Joint tenancy property passes by survivorship, NOT through will
                - Only tenancy in common shares can be bequeathed in will
                - Foreign property may require separate will in that jurisdiction
                - Land held under different title systems: Torrens (Peninsular), Native (Sabah/Sarawak)
                """
            },
            {
                "category": "financial_assets",
                "content": """
                FINANCIAL ASSETS:
                
                1. BANK ACCOUNTS
                   - Can use general or specific descriptions
                   - General: "all my bank accounts with [BANK NAME]"
                   - Specific: "my account number [NUMBER] with [BANK]"
                   - Joint accounts pass by survivorship unless tenancy in common
                
                2. EPF/KWSP (Employees Provident Fund)
                   - EPF has SEPARATE nomination system under EPF Act 1991
                   - Will CANNOT override EPF nominations
                   - Will provisions are backup only if no EPF nomination exists
                   - MANDATORY NOTICE: "EPF requires separate nomination at kwsp.gov.my"
                   
                3. INVESTMENTS
                   - Stocks and shares in public companies
                   - Unit trusts and mutual funds
                   - ASB/ASN/ASNB funds (Malaysian government unit trusts)
                   - Fixed deposits
                   - Bonds and securities
                
                4. INSURANCE POLICIES
                   - Policy nominations override will provisions
                   - Will clauses are backup for policies without nominations
                   - Include policy numbers and insurance companies
                
                SAMPLE CLAUSE:
                "I give all my bank accounts, deposits, and cash holdings with [BANK NAME]
                to my [RELATIONSHIP] [NAME] (NRIC No. [NUMBER])."
                """
            },
            {
                "category": "business_interests",
                "content": """
                BUSINESS INTERESTS:
                
                REQUIRED INFORMATION:
                - Company name
                - Registration number (SSM format for Malaysian companies)
                - Ownership percentage
                - Business type (sole proprietor/partnership/Sdn Bhd/Berhad)
                - Current valuation (estimated)
                
                SUCCESSION OPTIONS:
                1. Direct transfer to beneficiary
                2. Continuation with trustee management and gradual transfer
                3. Sale to existing partners or third parties
                4. Professional management with eventual transfer
                
                CONSIDERATIONS:
                - Check for shareholder agreements (may restrict transfer)
                - Consider buy-sell agreements with business partners
                - May need business valuation
                - Trustee may need power to continue operating business
                - Consider training period for beneficiary
                
                SAMPLE DIRECT TRANSFER CLAUSE:
                "I give all my shares and interests in [COMPANY NAME] (Company No. [NUMBER])
                comprising [PERCENTAGE]% shareholding, together with all goodwill, intellectual property,
                and business assets, to my [RELATIONSHIP] [NAME] (NRIC No. [NUMBER]), absolutely."
                """
            },
            {
                "category": "personal_property",
                "content": """
                PERSONAL PROPERTY:
                
                1. VEHICLES
                   - Registration number
                   - Make and model
                   - Transfer requires: Grant of Probate, JPJ procedures, Puspakom inspection
                   
                2. JEWELRY AND VALUABLES
                   - Specific descriptions helpful (e.g., "22-karat gold necklace with ruby pendant")
                   - Location if in safe deposit box
                   - Photographs or appraisals for high-value items
                   
                3. HOUSEHOLD ITEMS
                   - Can use general description: "all furniture, household effects, personal belongings"
                   - Or specific items: "my Steinway piano", "my art collection"
                   
                4. DIGITAL ASSETS (increasingly important)
                   - Cryptocurrency holdings (Bitcoin, Ethereum, etc.)
                   - NFTs and digital tokens
                   - Domain names
                   - Social media accounts
                   - Online business accounts
                   - Digital storage accounts
                   - NOTE: Provide location of access credentials
                   - Consider appointing "Digital Executor"
                   - Cryptocurrency regulated by Securities Commission Malaysia
                
                SAMPLE CLAUSES:
                "I give my motor vehicle registration number [NUMBER] to my [RELATIONSHIP] [NAME]."
                
                "I give all my digital assets including cryptocurrency holdings, domain names,
                and online accounts to [NAME], with access credentials stored at [LOCATION]."
                """
            }
        ]
    
    @staticmethod
    def get_distribution_clauses() -> List[Dict[str, str]]:
        """
        Standard distribution clause templates used in Malaysian wills.
        
        These templates use legally precise language from actual Malaysian wills.
        The RAG system can retrieve and adapt these based on user requirements.
        """
        return [
            {
                "category": "specific_bequest",
                "content": """
                SPECIFIC BEQUEST CLAUSES:
                
                Format: "I give, devise and bequeath [PROPERTY] to [BENEFICIARY] (NRIC No. [NUMBER]), [CONDITIONS]."
                
                EXAMPLES:
                
                Simple absolute gift:
                "I give, devise and bequeath my property at 123 Jalan Merdeka, Kuala Lumpur,
                held under Title No. HS(D) 12345, to my daughter, Siti Abdullah (NRIC No. 850315-08-5432), absolutely."
                
                Multiple beneficiaries with equal shares:
                "I give my property at [ADDRESS] to my children Ahmad bin Hassan (NRIC No. 780101-14-1234)
                and Fatimah binti Hassan (NRIC No. 820505-14-5678) in equal shares as tenants in common."
                
                Conditional bequest:
                "I give my property at [ADDRESS] to my son [NAME] PROVIDED THAT he completes his
                university degree by age 25, otherwise to my daughter [NAME]."
                
                With contingency:
                "I give [PROPERTY] to my sister [NAME] (NRIC No. [NUMBER]). IF my said sister shall
                predecease me, THEN I give the said property to her children in equal shares."
                """
            },
            {
                "category": "residuary_clause",
                "content": """
                RESIDUARY ESTATE CLAUSE (Mandatory - Critical):
                
                Purpose: Catches all property not specifically mentioned, prevents partial intestacy
                
                SIMPLE RESIDUARY:
                "I give all the rest, residue, and remainder of my estate, both real and personal,
                of whatsoever nature and wheresoever situated, which I may die possessed of or
                entitled to, and not hereby or by any codicil hereto otherwise specifically disposed of,
                including any lapsed or void gifts, to my [RELATIONSHIP] [NAME] (NRIC No. [NUMBER]), absolutely."
                
                PERCENTAGE RESIDUARY:
                "I give the residue of my estate as follows:
                (a) 50% to my wife [NAME] (NRIC No. [NUMBER]);
                (b) 25% to my son [NAME] (NRIC No. [NUMBER]);
                (c) 25% to my daughter [NAME] (NRIC No. [NUMBER])."
                
                EQUAL SHARES WITH PER STIRPES:
                "I give the residue of my estate to my children [LIST NAMES WITH NRIC] in equal shares
                absolutely, per stirpes, such that if any child predeceases me leaving issue,
                such issue shall take their parent's share in equal portions."
                
                Why residuary clause is essential:
                - Covers assets acquired after will execution
                - Catches lapsed gifts (beneficiary predeceases testator)
                - Catches void gifts (witness-beneficiary violations)
                - Prevents partial intestacy under Distribution Act 1958
                """
            },
            {
                "category": "trust_provisions",
                "content": """
                TRUST PROVISIONS FOR MINORS:
                
                Basic Trust for Minor:
                "I give [PROPERTY/AMOUNT] to my Trustee UPON TRUST for my [RELATIONSHIP] [NAME]
                (NRIC No. [NUMBER]) PROVIDED THAT:
                (a) My Trustee shall hold the said property until [BENEFICIARY] attains the age of [AGE] years;
                (b) Until that time, my Trustee may apply income and/or capital for maintenance, education,
                    healthcare, and general benefit;
                (c) My Trustee shall pay monthly allowances as deemed reasonable for living expenses;
                (d) Upon attaining [AGE] years, the entire trust property shall vest absolutely in [BENEFICIARY];
                (e) If [BENEFICIARY] dies before attaining [AGE] years, the trust property shall pass to [ALTERNATE]."
                
                Common vesting ages:
                - 18 years: age of majority in Malaysia
                - 21 years: traditional coming of age
                - 25 years: for larger estates or concerns about maturity
                
                Discretionary Trust:
                "My Trustee shall have absolute discretion to pay or apply capital and income for the
                benefit of [BENEFICIARY] as the Trustee deems appropriate for maintenance, education,
                advancement, or benefit, without being required to maintain equality."
                """
            },
            {
                "category": "contingency_clauses",
                "content": """
                CONTINGENCY CLAUSES:
                
                Purpose: Specify what happens if beneficiary predeceases testator
                
                Simple Contingency:
                "IF [PRIMARY BENEFICIARY] shall predecease me, THEN I give the said [PROPERTY]
                to [ALTERNATE BENEFICIARY]."
                
                Multiple Layers:
                "IF my sister [NAME] shall predecease me, THEN to her children in equal shares.
                IF she shall predecease me leaving no children, THEN to my brother [NAME]."
                
                Survivorship Period:
                "If any beneficiary shall die within 30 days of my death, such beneficiary shall be
                deemed to have predeceased me and the provisions relating to that beneficiary shall
                take effect accordingly."
                
                Purpose of survivorship period:
                - Prevents assets passing to beneficiary's estate then redistributing
                - Saves double administration costs
                - Common in accidents where testator and beneficiary die close together
                """
            }
        ]
    
    @staticmethod
    def get_validation_rules() -> List[Dict[str, str]]:
        """
        Validation rules for Malaysian will generation.
        
        These rules catch common errors that would invalidate a will
        or cause problems during probate. The system should check these
        before generating the final document.
        """
        return [
            {
                "category": "testator_validation",
                "content": """
                TESTATOR VALIDATION RULES:
                
                1. AGE VALIDATION
                   - Must be 18+ years (21+ in Sabah)
                   - Calculate from NRIC or date of birth
                   - BLOCK generation if under minimum age
                   
                2. NRIC FORMAT VALIDATION
                   - Must be 12 digits
                   - Format: YYMMDD-PB-###G
                   - YY = year, MM = month, DD = day
                   - PB = place of birth code
                   - ### = sequence number
                   - G = gender (odd = male, even = female)
                   
                3. RELIGION VALIDATION
                   - Muslim = STOP and recommend Wasiat instead
                   - Display message: "As a Muslim, you should create a Wasiat under Syariah law,
                     not a will. Only 1/3 of your estate can be bequeathed; remainder follows Faraid."
                   - Non-Muslim = proceed with standard will
                   
                4. REQUIRED FIELDS
                   - Full legal name (as per NRIC/identity document)
                   - NRIC number
                   - Complete address (street, city, postcode, state)
                   - Marital status
                   - Religion
                """
            },
            {
                "category": "witness_validation",
                "content": """
                WITNESS VALIDATION RULES (Critical for Validity):
                
                1. MINIMUM COUNT
                   - Must have exactly 2 witnesses (no more, no less for validity)
                   
                2. BENEFICIARY CONFLICT CHECK (Section 9 Wills Act 1959)
                   - Witness NRIC ≠ ANY beneficiary NRIC
                   - Witness NRIC ≠ spouse of ANY beneficiary
                   - If conflict detected: ALERT "CRITICAL ERROR: [WITNESS NAME] is a beneficiary.
                     Their gift will be VOID. Choose different witnesses."
                   
                3. EXECUTOR AS WITNESS
                   - Executor CAN witness IF NOT also a beneficiary
                   - Check: Is executor also receiving any bequest? If yes, WARN
                   
                4. AGE VALIDATION
                   - Preferably 21+ years
                   - Minimum 18 years
                   - Warn if under 21
                   
                5. REQUIRED DETAILS FOR EACH WITNESS
                   - Full legal name
                   - NRIC number (12-digit Malaysian format)
                   - Complete address
                   - Cannot be same person (check NRIC uniqueness)
                """
            },
            {
                "category": "distribution_validation",
                "content": """
                DISTRIBUTION VALIDATION RULES:
                
                1. PERCENTAGE TOTALS
                   - If using percentage distribution, must total exactly 100%
                   - Check residuary clause if specific bequests don't total 100%
                   
                2. BENEFICIARY COMPLETENESS
                   - Each beneficiary must have: name, NRIC/passport, relationship, address
                   - Minor beneficiaries (under 18) must trigger guardian appointment
                   
                3. RESIDUARY CLAUSE CHECK
                   - MUST have residuary clause (critical)
                   - If missing, ALERT: "No residuary clause detected. Any unmentioned assets
                     will pass under intestacy rules (Distribution Act 1958). Add residuary clause."
                   
                4. CONTINGENCY CHECK
                   - Recommend contingency beneficiaries for all major bequests
                   - Warn if beneficiary might predecease without alternate named
                   
                5. ASSET CONFLICTS
                   - Joint tenancy assets cannot be bequeathed (pass by survivorship)
                   - EPF/KWSP requires separate nomination
                   - Insurance policy nominations override will
                """
            },
            {
                "category": "structural_validation",
                "content": """
                STRUCTURAL VALIDATION RULES:
                
                Required sections in order:
                1. Opening declaration (testator identification)
                2. Revocation clause
                3. Executor appointment
                4. Debts and expenses clause
                5. Distribution clauses (specific bequests)
                6. Residuary clause
                7. Guardian appointment (if minors exist)
                8. Testimonium clause
                9. Signature blocks (testator + 2 witnesses)
                
                Missing section check:
                - If testator has children under 18, MUST have guardian appointment
                - If any beneficiary is minor, SHOULD have trust provisions
                - MUST have executor appointment (1-4 executors)
                - MUST have at least one alternate executor (recommended)
                
                Language check:
                - Distribution clauses use "I give, devise and bequeath"
                - Absolute gifts use "absolutely"
                - Trust provisions use "UPON TRUST"
                - Conditional gifts use "PROVIDED THAT"
                """
            }
        ]
    
    @staticmethod
    def create_all_documents() -> List[Document]:
        """
        Creates all knowledge base documents for RAG system.
        
        This method combines all our structured knowledge into LangChain Document
        objects with metadata. The RAG system will embed these documents and
        retrieve relevant ones when generating will clauses or validating inputs.
        """
        all_docs = []
        
        # Add legal requirements
        for doc in MalaysianWillKnowledgeBase.get_legal_requirements():
            all_docs.append(
                Document(
                    page_content=doc["content"],
                    metadata={"category": doc["category"], "type": "legal_requirement"}
                )
            )
        
        # Add asset categories
        for doc in MalaysianWillKnowledgeBase.get_asset_categories():
            all_docs.append(
                Document(
                    page_content=doc["content"],
                    metadata={"category": doc["category"], "type": "asset_category"}
                )
            )
        
        # Add distribution clauses
        for doc in MalaysianWillKnowledgeBase.get_distribution_clauses():
            all_docs.append(
                Document(
                    page_content=doc["content"],
                    metadata={"category": doc["category"], "type": "clause_template"}
                )
            )
        
        # Add validation rules
        for doc in MalaysianWillKnowledgeBase.get_validation_rules():
            all_docs.append(
                Document(
                    page_content=doc["content"],
                    metadata={"category": doc["category"], "type": "validation_rule"}
                )
            )
        
        return all_docs

# Create knowledge base documents
kb_documents = MalaysianWillKnowledgeBase.create_all_documents()
print(f"✅ Knowledge base created with {len(kb_documents)} documents")
print(f"   - Legal requirements: {len(MalaysianWillKnowledgeBase.get_legal_requirements())}")
print(f"   - Asset categories: {len(MalaysianWillKnowledgeBase.get_asset_categories())}")
print(f"   - Distribution clauses: {len(MalaysianWillKnowledgeBase.get_distribution_clauses())}")
print(f"   - Validation rules: {len(MalaysianWillKnowledgeBase.get_validation_rules())}")
