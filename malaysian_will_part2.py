"""
Malaysian Will Generation System with RAG Architecture
Part 2: Vector Store, RAG System, and Data Validation

This part builds the retrieval system and validates user inputs
against Malaysian legal requirements.
"""

# ============================================================================
# CELL 5: Vector Store Creation and RAG Setup
# ============================================================================

class WillRAGSystem:
    """
    Retrieval Augmented Generation system for Malaysian will generation.
    
    This class handles:
    1. Creating embeddings from our knowledge base
    2. Storing them in a vector database (Chroma)
    3. Retrieving relevant legal information when needed
    4. Generating will clauses using the LLM with retrieved context
    """
    
    def __init__(self, config: WillGeneratorConfig):
        """
        Initialize the RAG system.
        
        Parameters:
        -----------
        config : WillGeneratorConfig
            Configuration object with LLM and embedding settings
        """
        self.config = config
        self.embeddings = config.get_embeddings()
        self.llm = config.get_llm()
        self.vector_store = None
        self.retrieval_qa = None
        
        print("🔧 Initializing RAG system...")
        print(f"   Using embeddings: {config.embedding_model}")
        print(f"   Using LLM: {config.model_name} via {config.llm_provider}")
    
    def create_vector_store(self, documents: List[Document], force_recreate: bool = False):
        """
        Create vector store from knowledge base documents.
        
        This method:
        1. Takes our knowledge base documents
        2. Converts them to numerical embeddings (vectors)
        3. Stores them in Chroma database for fast similarity search
        
        The embeddings capture semantic meaning, so when we ask
        "What are the witness requirements?", it finds relevant documents
        even if the exact words don't match.
        
        Parameters:
        -----------
        documents : List[Document]
            Knowledge base documents to embed
        force_recreate : bool
            If True, delete existing store and create new one
        """
        store_path = str(self.config.vector_store_path)
        
        # Check if vector store already exists
        if os.path.exists(store_path) and not force_recreate:
            print(f"📚 Loading existing vector store from {store_path}")
            self.vector_store = Chroma(
                persist_directory=store_path,
                embedding_function=self.embeddings
            )
        else:
            if force_recreate and os.path.exists(store_path):
                print(f"🗑️  Deleting existing vector store...")
                import shutil
                shutil.rmtree(store_path)
            
            print(f"📚 Creating new vector store with {len(documents)} documents...")
            
            # Split documents into smaller chunks for better retrieval
            # We use small chunks (500 chars) because legal text is dense
            # and we want precise retrieval of specific requirements
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            print(f"   Split into {len(split_docs)} chunks for embedding")
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=store_path
            )
            
            # Persist to disk
            self.vector_store.persist()
            print(f"   ✅ Vector store created and saved to {store_path}")
        
        return self.vector_store
    
    def setup_retrieval_qa(self, search_type: str = "similarity", k: int = 4):
        """
        Setup the QA chain for retrieving and generating responses.
        
        This creates a question-answering system that:
        1. Takes a question (e.g., "What are witness requirements?")
        2. Retrieves the k most relevant documents from vector store
        3. Passes them to the LLM as context
        4. Generates an answer based on the retrieved information
        
        Parameters:
        -----------
        search_type : str
            Type of search ("similarity" or "mmr" - maximum marginal relevance)
        k : int
            Number of documents to retrieve (default 4 is good balance)
        """
        if self.vector_store is None:
            raise ValueError("Vector store not created. Call create_vector_store() first.")
        
        # Create retriever with specified search parameters
        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        
        # Create custom prompt for Malaysian will generation
        template = """You are an expert in Malaysian will drafting under the Wills Act 1959.
        Use the following legal information to answer the question accurately and precisely.
        If you're not sure about something, say so rather than making up information.
        Always cite the relevant section of the Wills Act when applicable.
        
        Context from Malaysian law and will requirements:
        {context}
        
        Question: {question}
        
        Detailed Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the RetrievalQA chain
        self.retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" puts all retrieved docs into prompt
            retriever=retriever,
            return_source_documents=True,  # Return sources for transparency
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        
        print(f"✅ QA system ready (retrieving {k} documents per query)")
        return self.retrieval_qa
    
    def query(self, question: str, return_sources: bool = False):
        """
        Query the RAG system.
        
        This is the main interface for asking questions about Malaysian will law
        and retrieving relevant legal information.
        
        Parameters:
        -----------
        question : str
            The question to ask
        return_sources : bool
            If True, also return the source documents used
        
        Returns:
        --------
        dict or str
            The answer (and optionally source documents)
        """
        if self.retrieval_qa is None:
            raise ValueError("QA system not setup. Call setup_retrieval_qa() first.")
        
        result = self.retrieval_qa({"query": question})
        
        if return_sources:
            return {
                "answer": result["result"],
                "sources": [doc.page_content for doc in result["source_documents"]]
            }
        else:
            return result["result"]

# Initialize RAG system
print("\n" + "="*70)
print("SETTING UP RAG SYSTEM")
print("="*70)

rag_system = WillRAGSystem(config)

# Create vector store from knowledge base
rag_system.create_vector_store(kb_documents, force_recreate=False)

# Setup QA chain
rag_system.setup_retrieval_qa(k=4)

# Test the system
print("\n📝 Testing RAG system...")
test_query = "What are the witness requirements for a valid Malaysian will?"
test_answer = rag_system.query(test_query)
print(f"\nQ: {test_query}")
print(f"A: {test_answer[:200]}...")  # Print first 200 chars

# ============================================================================
# CELL 6: Data Models for Validation
# ============================================================================

class PersonInfo(BaseModel):
    """Represents a person (testator, beneficiary, executor, witness)."""
    
    full_name: str = Field(..., description="Full legal name as per NRIC")
    nric: str = Field(..., description="Malaysian NRIC number (12 digits)")
    address: str = Field(..., description="Complete residential address")
    relationship: Optional[str] = Field(None, description="Relationship to testator")
    
    @validator('nric')
    def validate_nric(cls, v):
        """Validate Malaysian NRIC format: YYMMDD-PB-###G"""
        # Remove any dashes or spaces
        clean_nric = re.sub(r'[-\s]', '', v)
        
        if len(clean_nric) != 12:
            raise ValueError(f"NRIC must be 12 digits, got {len(clean_nric)}")
        
        if not clean_nric.isdigit():
            raise ValueError("NRIC must contain only digits")
        
        # Validate date portion (first 6 digits)
        year = int(clean_nric[0:2])
        month = int(clean_nric[2:4])
        day = int(clean_nric[4:6])
        
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month in NRIC: {month}")
        
        if day < 1 or day > 31:
            raise ValueError(f"Invalid day in NRIC: {day}")
        
        return clean_nric
    
    def format_nric(self) -> str:
        """Format NRIC with dashes: YYMMDD-PB-###G"""
        nric = self.nric
        return f"{nric[0:6]}-{nric[6:8]}-{nric[8:12]}"
    
    def get_age(self) -> int:
        """Calculate age from NRIC."""
        year = int(self.nric[0:2])
        # Determine century (before 2000 or after)
        current_year = datetime.now().year % 100
        if year > current_year:
            full_year = 1900 + year
        else:
            full_year = 2000 + year
        
        month = int(self.nric[2:4])
        day = int(self.nric[4:6])
        
        birth_date = datetime(full_year, month, day)
        today = datetime.now()
        
        age = today.year - birth_date.year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1
        
        return age


class TestatorInfo(PersonInfo):
    """Testator-specific information."""
    
    date_of_birth: Optional[str] = Field(None, description="Date of birth if NRIC unclear")
    religion: str = Field(..., description="Religion (for routing Muslims to Wasiat)")
    marital_status: str = Field(..., description="Single/Married/Divorced/Widowed")
    
    @validator('religion')
    def check_religion(cls, v):
        """Warn if Muslim (should use Wasiat instead of Will)."""
        if v.lower() in ['muslim', 'islam']:
            print("\n⚠️  WARNING: As a Muslim, you should create a Wasiat under Syariah law.")
            print("   Only 1/3 of your estate can be bequeathed; remainder follows Faraid.")
            print("   Consider consulting a Syariah advisor.")
        return v
    
    def validate_age(self) -> Tuple[bool, str]:
        """Validate minimum age requirement."""
        age = self.get_age()
        
        if age < 18:
            return False, f"Testator must be 18+ years old. Current age: {age}"
        
        if age < 21:
            return True, f"Age {age} is valid for Peninsular Malaysia. Note: Sabah requires 21+"
        
        return True, f"Age {age} - valid for will creation"


class BeneficiaryInfo(PersonInfo):
    """Beneficiary-specific information."""
    
    is_minor: bool = Field(default=False, description="Is beneficiary under 18?")
    distribution_type: str = Field(..., description="specific/percentage/equal/residuary")
    distribution_value: Optional[str] = Field(None, description="Specific asset or percentage")
    contingent_beneficiary: Optional[str] = Field(None, description="Alternate if predeceases")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Auto-detect if minor based on age
        if self.get_age() < 18:
            self.is_minor = True


class AssetInfo(BaseModel):
    """Represents an asset to be distributed."""
    
    asset_type: str = Field(..., description="real_property/bank_account/investment/business/vehicle/personal/digital")
    description: str = Field(..., description="Detailed description of asset")
    value: Optional[float] = Field(None, description="Estimated value in RM")
    specific_details: Dict[str, str] = Field(default_factory=dict, description="Type-specific details")
    beneficiary_nric: str = Field(..., description="NRIC of beneficiary for this asset")
    
    @validator('asset_type')
    def validate_asset_type(cls, v):
        """Ensure valid asset type."""
        valid_types = [
            'real_property', 'bank_account', 'investment', 'epf_kwsp',
            'business', 'vehicle', 'jewelry', 'personal', 'digital', 'insurance'
        ]
        if v not in valid_types:
            raise ValueError(f"Asset type must be one of: {', '.join(valid_types)}")
        return v
    
    def get_required_details(self) -> List[str]:
        """Return required specific details based on asset type."""
        requirements = {
            'real_property': ['address', 'title_number', 'ownership_type'],
            'bank_account': ['bank_name', 'account_type'],
            'investment': ['investment_type', 'institution'],
            'epf_kwsp': ['account_number'],
            'business': ['company_name', 'registration_number', 'ownership_percentage'],
            'vehicle': ['registration_number', 'make_model'],
            'digital': ['asset_description', 'access_location']
        }
        return requirements.get(self.asset_type, [])
    
    def validate_specific_details(self) -> Tuple[bool, str]:
        """Check if all required specific details are provided."""
        required = self.get_required_details()
        missing = [field for field in required if field not in self.specific_details]
        
        if missing:
            return False, f"Missing required details for {self.asset_type}: {', '.join(missing)}"
        
        return True, "All required details provided"


class WillData(BaseModel):
    """Complete will information."""
    
    testator: TestatorInfo
    executors: List[PersonInfo] = Field(..., min_items=1, max_items=4)
    witnesses: List[PersonInfo] = Field(..., min_items=2, max_items=2)
    beneficiaries: List[BeneficiaryInfo]
    assets: List[AssetInfo] = Field(default_factory=list)
    guardians: List[PersonInfo] = Field(default_factory=list)
    has_minor_children: bool = Field(default=False)
    special_instructions: Optional[str] = Field(None)
    
    @validator('witnesses')
    def validate_witnesses(cls, v, values):
        """Critical validation: witnesses cannot be beneficiaries."""
        if 'beneficiaries' not in values:
            return v
        
        beneficiary_nrics = {b.nric for b in values['beneficiaries']}
        
        for witness in v:
            if witness.nric in beneficiary_nrics:
                raise ValueError(
                    f"CRITICAL ERROR: {witness.full_name} is both a witness and beneficiary. "
                    f"Under Section 9 of Wills Act 1959, their gift will be VOID. "
                    f"Choose different witnesses."
                )
        
        return v
    
    @validator('guardians')
    def validate_guardians(cls, v, values):
        """If has minor children, must have guardians."""
        if values.get('has_minor_children', False) and len(v) == 0:
            raise ValueError("Must appoint at least one guardian for minor children")
        return v
    
    def validate_complete(self) -> Tuple[bool, List[str]]:
        """Run all validation checks and return errors."""
        errors = []
        
        # Validate testator age
        valid, msg = self.testator.validate_age()
        if not valid:
            errors.append(msg)
        
        # Check for executor-beneficiary-witness conflicts
        executor_nrics = {e.nric for e in self.executors}
        beneficiary_nrics = {b.nric for b in self.beneficiaries}
        witness_nrics = {w.nric for w in self.witnesses}
        
        # Executors can be beneficiaries OR witnesses, but not both
        executor_both = executor_nrics & beneficiary_nrics & witness_nrics
        if executor_both:
            errors.append(
                f"Executors cannot be BOTH beneficiaries AND witnesses: "
                f"{[e.full_name for e in self.executors if e.nric in executor_both]}"
            )
        
        # Check asset-specific details
        for asset in self.assets:
            valid, msg = asset.validate_specific_details()
            if not valid:
                errors.append(msg)
        
        # Check for minor beneficiaries without trust provisions
        minor_beneficiaries = [b for b in self.beneficiaries if b.is_minor]
        if minor_beneficiaries:
            errors.append(
                f"WARNING: {len(minor_beneficiaries)} minor beneficiary(ies) detected. "
                f"Consider adding trust provisions with vesting age."
            )
        
        return len(errors) == 0, errors

# ============================================================================
# CELL 7: Test Data Validation
# ============================================================================

print("\n" + "="*70)
print("TESTING DATA VALIDATION")
print("="*70)

# Create test testator
test_testator = TestatorInfo(
    full_name="Ahmad bin Hassan",
    nric="780101-14-1234",
    address="123 Jalan Merdeka, 50000 Kuala Lumpur",
    religion="Non-Muslim",
    marital_status="Married"
)

print(f"\n✅ Testator: {test_testator.full_name}")
print(f"   NRIC: {test_testator.format_nric()}")
print(f"   Age: {test_testator.get_age()} years")
valid, msg = test_testator.validate_age()
print(f"   Age validation: {msg}")

# Create test beneficiaries
test_beneficiaries = [
    BeneficiaryInfo(
        full_name="Siti binti Ahmad",
        nric="800505-14-5678",
        address="123 Jalan Merdeka, 50000 KL",
        relationship="Wife",
        distribution_type="percentage",
        distribution_value="50%"
    ),
    BeneficiaryInfo(
        full_name="Fatimah binti Ahmad", 
        nric="100303-14-9012",
        address="123 Jalan Merdeka, 50000 KL",
        relationship="Daughter",
        distribution_type="percentage",
        distribution_value="50%"
    )
]

print(f"\n✅ Beneficiaries: {len(test_beneficiaries)}")
for b in test_beneficiaries:
    minor_status = "MINOR" if b.is_minor else "Adult"
    print(f"   - {b.full_name} ({b.relationship}) - {minor_status}")

print("\n✅ Data validation system ready!")
print("   All Pydantic models created with automatic validation")
print("   NRIC format validation active")
print("   Witness-beneficiary conflict detection enabled")
