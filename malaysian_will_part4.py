"""
Malaysian Will Generation System with RAG Architecture
Part 4: Complete Working Example and Interactive Questionnaire

This demonstrates the full workflow from data collection to will generation.
"""

# ============================================================================
# CELL 9: Interactive Questionnaire System
# ============================================================================

class WillQuestionnaire:
    """
    Interactive questionnaire system for collecting will information.
    
    This mimics the approach used by Malaysian will writing services like
    Rockwills, CreateWills.my, and NobleWills - guiding users through
    structured questions to gather all necessary information.
    """
    
    def __init__(self, rag_system: WillRAGSystem):
        """
        Initialize questionnaire with RAG system for contextual help.
        
        The RAG system can provide explanations and examples when users
        need help understanding what information to provide.
        """
        self.rag = rag_system
        self.responses = {}
    
    def ask_with_help(self, question: str, help_topic: Optional[str] = None) -> str:
        """
        Ask a question with optional RAG-powered help.
        
        In a real application, this would be a nice UI. For Jupyter,
        we'll use simple input() with the option to type 'help' for assistance.
        """
        print(f"\n{question}")
        if help_topic:
            print(f"(Type 'help' for more information about {help_topic})")
        
        response = input("> ").strip()
        
        if response.lower() == 'help' and help_topic:
            help_text = self.rag.query(f"Explain {help_topic} for Malaysian wills")
            print(f"\n📚 Help: {help_text}\n")
            response = input("> ").strip()
        
        return response
    
    def collect_testator_info(self) -> TestatorInfo:
        """Collect testator information through guided questions."""
        print("\n" + "="*70)
        print("SECTION 1: YOUR INFORMATION (TESTATOR)")
        print("="*70)
        print("\nLet's start with your basic information.")
        
        full_name = input("\nYour full legal name (as per NRIC): ").strip()
        
        nric = input("Your NRIC number (12 digits, e.g., 780101-14-1234): ").strip()
        
        address = input("Your complete residential address: ").strip()
        
        religion = input("Your religion (important for routing): ").strip()
        
        if religion.lower() in ['muslim', 'islam']:
            print("\n⚠️  IMPORTANT: As a Muslim, you should create a Wasiat, not a Will.")
            print("   Only 1/3 of your estate can be bequeathed in a Wasiat.")
            print("   The remaining 2/3 must follow Faraid (Islamic inheritance law).")
            proceed = input("\nDo you want to proceed with a standard Will anyway? (yes/no): ")
            if proceed.lower() != 'yes':
                raise ValueError("Please consult a Syariah advisor for Wasiat preparation.")
        
        marital_status = input("Marital status (Single/Married/Divorced/Widowed): ").strip()
        
        return TestatorInfo(
            full_name=full_name,
            nric=nric,
            address=address,
            religion=religion,
            marital_status=marital_status
        )
    
    def collect_executor_info(self) -> List[PersonInfo]:
        """Collect executor information."""
        print("\n" + "="*70)
        print("SECTION 2: EXECUTOR APPOINTMENT")
        print("="*70)
        print("\nYour executor will manage your estate after death.")
        print("You must appoint 1-4 executors. We recommend at least 2 (primary + backup).")
        
        help_text = self.rag.query("What does an executor do and who should I choose?")
        print(f"\n📚 {help_text[:300]}...")
        
        executors = []
        
        while len(executors) < 4:
            if len(executors) == 0:
                print("\n--- Primary Executor ---")
            else:
                print(f"\n--- Executor #{len(executors) + 1} (Alternate) ---")
                add_more = input("Add another executor? (yes/no): ").strip()
                if add_more.lower() != 'yes':
                    break
            
            name = input("Full name: ").strip()
            nric = input("NRIC number: ").strip()
            address = input("Complete address: ").strip()
            relationship = input("Relationship to you (e.g., Spouse, Son, Daughter, Friend): ").strip()
            
            executors.append(PersonInfo(
                full_name=name,
                nric=nric,
                address=address,
                relationship=relationship
            ))
        
        return executors
    
    def collect_beneficiary_info(self) -> List[BeneficiaryInfo]:
        """Collect beneficiary information."""
        print("\n" + "="*70)
        print("SECTION 3: BENEFICIARIES")
        print("="*70)
        print("\nWho should receive your assets?")
        
        beneficiaries = []
        
        while True:
            print(f"\n--- Beneficiary #{len(beneficiaries) + 1} ---")
            
            name = input("Full name: ").strip()
            nric = input("NRIC or passport number: ").strip()
            address = input("Complete address: ").strip()
            relationship = input("Relationship to you: ").strip()
            
            print("\nHow should this beneficiary receive their share?")
            print("1. Equal share with others (divide residuary estate equally)")
            print("2. Specific percentage of residuary estate")
            print("3. Specific assets only")
            print("4. Residuary beneficiary (gets everything not specifically mentioned)")
            
            dist_choice = input("Choose (1-4): ").strip()
            
            if dist_choice == "1":
                dist_type = "equal"
                dist_value = None
            elif dist_choice == "2":
                dist_type = "percentage"
                dist_value = input("Percentage (e.g., 50%): ").strip()
            elif dist_choice == "3":
                dist_type = "specific"
                dist_value = "See asset section"
            else:
                dist_type = "residuary"
                dist_value = None
            
            beneficiaries.append(BeneficiaryInfo(
                full_name=name,
                nric=nric,
                address=address,
                relationship=relationship,
                distribution_type=dist_type,
                distribution_value=dist_value
            ))
            
            add_more = input("\nAdd another beneficiary? (yes/no): ").strip()
            if add_more.lower() != 'yes':
                break
        
        return beneficiaries
    
    def collect_asset_info(self, beneficiaries: List[BeneficiaryInfo]) -> List[AssetInfo]:
        """Collect asset information."""
        print("\n" + "="*70)
        print("SECTION 4: ASSETS")
        print("="*70)
        print("\nLet's document your assets. You can skip this and use a general")
        print("residuary clause, or specify particular assets for particular beneficiaries.")
        
        skip = input("\nDo you want to list specific assets? (yes/no): ").strip()
        if skip.lower() != 'yes':
            return []
        
        assets = []
        
        # Create beneficiary lookup
        print("\nAvailable beneficiaries:")
        for i, b in enumerate(beneficiaries, start=1):
            print(f"{i}. {b.full_name} ({b.relationship})")
        
        while True:
            print(f"\n--- Asset #{len(assets) + 1} ---")
            print("\nAsset Types:")
            print("1. Real property (land/house/apartment)")
            print("2. Bank account")
            print("3. Investment (stocks/unit trusts/ASB)")
            print("4. EPF/KWSP")
            print("5. Business")
            print("6. Vehicle")
            print("7. Jewelry/valuables")
            print("8. Digital assets (crypto/domains/online accounts)")
            print("9. Other")
            
            asset_choice = input("Choose asset type (1-9): ").strip()
            
            asset_types = {
                "1": "real_property",
                "2": "bank_account",
                "3": "investment",
                "4": "epf_kwsp",
                "5": "business",
                "6": "vehicle",
                "7": "jewelry",
                "8": "digital",
                "9": "personal"
            }
            
            asset_type = asset_types.get(asset_choice, "personal")
            
            description = input("Brief description: ").strip()
            
            value_str = input("Estimated value in RM (optional, press Enter to skip): ").strip()
            value = float(value_str) if value_str else None
            
            # Collect type-specific details
            specific_details = {}
            
            if asset_type == "real_property":
                specific_details['address'] = input("Property address: ").strip()
                specific_details['title_number'] = input("Land title number: ").strip()
                specific_details['ownership_type'] = input("Ownership type (sole/joint): ").strip()
            
            elif asset_type == "bank_account":
                specific_details['bank_name'] = input("Bank name: ").strip()
                specific_details['account_type'] = input("Account type (savings/current): ").strip()
            
            elif asset_type == "vehicle":
                specific_details['registration_number'] = input("Registration number: ").strip()
                specific_details['make_model'] = input("Make and model: ").strip()
            
            elif asset_type == "business":
                specific_details['company_name'] = input("Company name: ").strip()
                specific_details['registration_number'] = input("SSM registration number: ").strip()
                specific_details['ownership_percentage'] = input("Your ownership %: ").strip()
            
            elif asset_type == "digital":
                specific_details['asset_description'] = description
                specific_details['access_location'] = input("Where are access credentials stored?: ").strip()
            
            # Choose beneficiary for this asset
            beneficiary_num = input("Which beneficiary receives this asset? (enter number): ").strip()
            beneficiary_nric = beneficiaries[int(beneficiary_num) - 1].nric
            
            assets.append(AssetInfo(
                asset_type=asset_type,
                description=description,
                value=value,
                specific_details=specific_details,
                beneficiary_nric=beneficiary_nric
            ))
            
            add_more = input("\nAdd another asset? (yes/no): ").strip()
            if add_more.lower() != 'yes':
                break
        
        return assets
    
    def collect_witness_info(self, beneficiaries: List[BeneficiaryInfo]) -> List[PersonInfo]:
        """Collect witness information with validation."""
        print("\n" + "="*70)
        print("SECTION 5: WITNESSES")
        print("="*70)
        print("\n⚠️  CRITICAL: Witnesses CANNOT be beneficiaries or spouses of beneficiaries!")
        print("   If they are, their gift becomes VOID under Section 9 of Wills Act 1959.")
        
        beneficiary_nrics = {b.nric for b in beneficiaries}
        witnesses = []
        
        for i in range(2):
            print(f"\n--- Witness #{i + 1} ---")
            
            while True:
                name = input("Full name: ").strip()
                nric = input("NRIC number: ").strip()
                
                # Check conflict
                if nric in beneficiary_nrics:
                    print(f"\n❌ ERROR: {name} is a beneficiary! Their gift will be VOID.")
                    print("   Please choose a different witness.")
                    continue
                
                address = input("Complete address: ").strip()
                
                witnesses.append(PersonInfo(
                    full_name=name,
                    nric=nric,
                    address=address
                ))
                break
        
        return witnesses
    
    def collect_guardian_info(self) -> Tuple[bool, List[PersonInfo]]:
        """Collect guardian information for minor children."""
        print("\n" + "="*70)
        print("SECTION 6: GUARDIAN FOR MINOR CHILDREN")
        print("="*70)
        
        has_minors = input("\nDo you have children under 18 years old? (yes/no): ").strip()
        
        if has_minors.lower() != 'yes':
            return False, []
        
        print("\nYou must appoint a guardian for your minor children.")
        print("This person will have custody and care until they turn 18.")
        
        guardians = []
        
        for i in range(2):
            if i == 0:
                print("\n--- Primary Guardian ---")
            else:
                add_backup = input("\nAdd backup guardian? (recommended, yes/no): ").strip()
                if add_backup.lower() != 'yes':
                    break
                print("\n--- Backup Guardian ---")
            
            name = input("Full name: ").strip()
            nric = input("NRIC number: ").strip()
            address = input("Complete address: ").strip()
            relationship = input("Relationship to you: ").strip()
            
            guardians.append(PersonInfo(
                full_name=name,
                nric=nric,
                address=address,
                relationship=relationship
            ))
        
        return True, guardians

# ============================================================================
# CELL 10: Complete Working Example
# ============================================================================

def run_complete_example():
    """
    Complete example showing the entire will generation workflow.
    
    This demonstrates:
    1. Using the interactive questionnaire
    2. Validating all inputs
    3. Generating the will document
    4. Saving to file
    5. Providing signing instructions
    """
    
    print("\n" + "="*70)
    print("MALAYSIAN WILL GENERATION SYSTEM")
    print("For Non-Muslims under Wills Act 1959")
    print("="*70)
    print("\nThis system will guide you through creating a legally valid will.")
    print("You can type 'help' at any question for more information.")
    print("\nIMPORTANT: This generates a draft will. We strongly recommend")
    print("having it reviewed by a qualified lawyer before signing.")
    
    proceed = input("\nReady to begin? (yes/no): ").strip()
    if proceed.lower() != 'yes':
        print("Questionnaire cancelled.")
        return
    
    # Initialize questionnaire
    questionnaire = WillQuestionnaire(rag_system)
    
    try:
        # Collect all information
        testator = questionnaire.collect_testator_info()
        executors = questionnaire.collect_executor_info()
        beneficiaries = questionnaire.collect_beneficiary_info()
        assets = questionnaire.collect_asset_info(beneficiaries)
        witnesses = questionnaire.collect_witness_info(beneficiaries)
        has_minors, guardians = questionnaire.collect_guardian_info()
        
        # Ask for special instructions
        print("\n" + "="*70)
        print("SECTION 7: SPECIAL INSTRUCTIONS (Optional)")
        print("="*70)
        print("\nAny special instructions? (funeral wishes, charitable donations, etc.)")
        special = input("Enter instructions or press Enter to skip: ").strip()
        special_instructions = special if special else None
        
        # Create WillData object
        will_data = WillData(
            testator=testator,
            executors=executors,
            witnesses=witnesses,
            beneficiaries=beneficiaries,
            assets=assets,
            guardians=guardians,
            has_minor_children=has_minors,
            special_instructions=special_instructions
        )
        
        # Validate
        print("\n" + "="*70)
        print("VALIDATING INFORMATION")
        print("="*70)
        
        valid, errors = will_data.validate_complete()
        
        if not valid:
            print("\n⚠️  Validation warnings/errors:")
            for error in errors:
                print(f"   - {error}")
            
            if any("ERROR" in error for error in errors):
                print("\n❌ Critical errors found. Cannot proceed.")
                return
            
            print("\nWarnings noted. Proceeding with will generation...")
        else:
            print("✅ All validations passed!")
        
        # Generate will
        print("\n" + "="*70)
        print("GENERATING WILL DOCUMENT")
        print("="*70)
        
        will_text = will_generator.generate_will(will_data)
        
        # Save will
        will_path = will_generator.save_will(will_text)
        
        # Generate signing instructions
        instructions = will_generator.generate_signing_instructions()
        instructions_path = will_generator.save_will(
            instructions,
            filename=f"SIGNING_INSTRUCTIONS_{testator.full_name.replace(' ', '_')}.txt"
        )
        
        # Display summary
        print("\n" + "="*70)
        print("✅ WILL GENERATION COMPLETE!")
        print("="*70)
        print(f"\nYour will has been saved to:")
        print(f"   📄 {will_path}")
        print(f"\nSigning instructions saved to:")
        print(f"   📋 {instructions_path}")
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Review the will carefully")
        print("2. Consider having it reviewed by a lawyer")
        print("3. Gather yourself and TWO witnesses (18+ years, not beneficiaries)")
        print("4. All three people must be present at the same time")
        print("5. Follow the signing instructions exactly")
        print("6. Store the original safely and inform your executor")
        print("\n7. IMPORTANT REMINDERS:")
        print("   - Marriage will REVOKE this will")
        print("   - Update EPF nominations separately at kwsp.gov.my")
        print("   - Update insurance policy nominations")
        print("   - Review will every 3-5 years or after major life changes")
        
        # Display preview
        print("\n" + "="*70)
        print("WILL PREVIEW (first 1000 characters)")
        print("="*70)
        print(will_text[:1000] + "...\n")
        
        return will_path, instructions_path
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# CELL 11: Alternative - Programmatic Example (No User Input)
# ============================================================================

def create_sample_will():
    """
    Create a sample will programmatically (no user input required).
    
    This is useful for:
    1. Testing the system
    2. Demonstrations
    3. Integration with web forms or other UIs
    """
    
    print("\n" + "="*70)
    print("CREATING SAMPLE WILL (Programmatic)")
    print("="*70)
    
    # Create sample data
    testator = TestatorInfo(
        full_name="Tan Wei Ming",
        nric="780505-10-1234",
        address="45 Jalan Bukit Bintang, 50200 Kuala Lumpur, Wilayah Persekutuan",
        religion="Buddhist",
        marital_status="Married"
    )
    
    executors = [
        PersonInfo(
            full_name="Lim Mei Ling",
            nric="800808-10-5678",
            address="45 Jalan Bukit Bintang, 50200 Kuala Lumpur",
            relationship="Wife"
        ),
        PersonInfo(
            full_name="Tan Ah Kow",
            nric="500303-10-9999",
            address="12 Jalan Gasing, 46000 Petaling Jaya, Selangor",
            relationship="Brother"
        )
    ]
    
    beneficiaries = [
        BeneficiaryInfo(
            full_name="Lim Mei Ling",
            nric="800808-10-5678",
            address="45 Jalan Bukit Bintang, 50200 KL",
            relationship="Wife",
            distribution_type="percentage",
            distribution_value="60%"
        ),
        BeneficiaryInfo(
            full_name="Tan Xiao Ming",
            nric="100202-10-1111",
            address="45 Jalan Bukit Bintang, 50200 KL",
            relationship="Son",
            distribution_type="percentage",
            distribution_value="40%"
        )
    ]
    
    assets = [
        AssetInfo(
            asset_type="real_property",
            description="Residential condominium",
            value=800000.0,
            specific_details={
                'address': "Unit 12-3, Menara Hartamas, Jalan Sri Hartamas 1, 50480 KL",
                'title_number': "HS(D) 12345 PT 67890",
                'ownership_type': "sole"
            },
            beneficiary_nric="800808-10-5678"
        ),
        AssetInfo(
            asset_type="bank_account",
            description="All accounts with Maybank",
            value=150000.0,
            specific_details={
                'bank_name': "Maybank Berhad",
                'account_type': "savings and current accounts"
            },
            beneficiary_nric="800808-10-5678"
        ),
        AssetInfo(
            asset_type="vehicle",
            description="Family car",
            value=80000.0,
            specific_details={
                'registration_number': "WXY 1234",
                'make_model': "Toyota Camry 2020"
            },
            beneficiary_nric="100202-10-1111"
        )
    ]
    
    witnesses = [
        PersonInfo(
            full_name="Wong Siew Lan",
            nric="750606-10-3333",
            address="78 Jalan Ampang, 50450 Kuala Lumpur"
        ),
        PersonInfo(
            full_name="Kumar Subramaniam",
            nric="820909-10-4444",
            address="23 Jalan Tun Razak, 50400 Kuala Lumpur"
        )
    ]
    
    guardians = [
        PersonInfo(
            full_name="Tan Ah Kow",
            nric="500303-10-9999",
            address="12 Jalan Gasing, 46000 Petaling Jaya",
            relationship="Uncle"
        )
    ]
    
    # Create WillData
    will_data = WillData(
        testator=testator,
        executors=executors,
        witnesses=witnesses,
        beneficiaries=beneficiaries,
        assets=assets,
        guardians=guardians,
        has_minor_children=True,
        special_instructions="I wish to be cremated and ashes scattered at sea."
    )
    
    # Validate
    valid, errors = will_data.validate_complete()
    print(f"\nValidation: {'✅ Passed' if valid else '⚠️  Warnings'}")
    if errors:
        for error in errors:
            print(f"   - {error}")
    
    # Generate will
    print("\nGenerating will document...")
    will_text = will_generator.generate_will(will_data)
    
    # Save
    will_path = will_generator.save_will(will_text)
    
    print(f"\n✅ Sample will generated successfully!")
    print(f"   Saved to: {will_path}")
    
    # Print full will
    print("\n" + "="*70)
    print("COMPLETE GENERATED WILL")
    print("="*70)
    print(will_text)
    print("="*70)
    
    return will_text

# ============================================================================
# Run the example
# ============================================================================

print("\n" + "="*70)
print("MALAYSIAN WILL GENERATION - READY")
print("="*70)
print("\nChoose an option:")
print("1. Interactive questionnaire (requires user input)")
print("2. Generate sample will (automatic, no input)")
print("\nFor Jupyter notebook, we'll generate a sample will automatically.")
print("In production, you would use option 1 for real users.\n")

# Generate sample will automatically for demonstration
sample_will_text = create_sample_will()

print("\n" + "="*70)
print("SYSTEM READY FOR PRODUCTION USE")
print("="*70)
print("\nTo use this system:")
print("\n1. For interactive use:")
print("   >>> run_complete_example()")
print("\n2. For web integration:")
print("   - Collect user data via web forms")
print("   - Create WillData object")
print("   - Call will_generator.generate_will(will_data)")
print("\n3. For API integration:")
print("   - Accept JSON input with all fields")
print("   - Validate using Pydantic models")
print("   - Return generated will as PDF/text")
print("\nAll modules are modular and can be customized!")
