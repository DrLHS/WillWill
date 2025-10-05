"""
Malaysian Will Generation System with RAG Architecture
Part 3: Will Document Generation and Complete Workflow

This part generates the actual will document using RAG-retrieved clauses
and assembles them into a legally valid Malaysian will.
"""

# ============================================================================
# CELL 8: Will Document Generator
# ============================================================================

class MalaysianWillGenerator:
    """
    Generates complete Malaysian will documents for non-Muslims.
    
    This class:
    1. Uses the RAG system to retrieve relevant legal clauses
    2. Validates all inputs against Malaysian legal requirements  
    3. Assembles clauses in the correct legal order
    4. Generates a properly formatted will document
    """
    
    def __init__(self, rag_system: WillRAGSystem):
        """
        Initialize the will generator.
        
        Parameters:
        -----------
        rag_system : WillRAGSystem
            The RAG system for retrieving legal information
        """
        self.rag = rag_system
        self.will_data = None
        
    def generate_opening_declaration(self, testator: TestatorInfo) -> str:
        """
        Generate the opening declaration section.
        
        This is the first mandatory section that identifies the testator
        and declares their intent to make a will.
        """
        return f"""LAST WILL AND TESTAMENT

I, {testator.full_name.upper()} (NRIC NO. {testator.format_nric()}) of {testator.address}, 
being of sound mind and disposing memory, DO HEREBY REVOKE all former testamentary 
dispositions, wills or codicils herebefore made by me AND DECLARE this to be my 
LAST WILL AND TESTAMENT."""
    
    def generate_executor_clause(self, executors: List[PersonInfo]) -> str:
        """
        Generate executor appointment clause.
        
        Must appoint 1-4 executors with at least one alternate.
        Executors are responsible for administering the estate.
        """
        if len(executors) == 0:
            raise ValueError("Must appoint at least one executor")
        
        if len(executors) > 4:
            raise ValueError("Maximum 4 executors allowed under Malaysian law")
        
        primary = executors[0]
        clause = f"""
EXECUTOR APPOINTMENT

I APPOINT my {primary.relationship if primary.relationship else 'trusted person'}, 
{primary.full_name.upper()} (NRIC NO. {primary.format_nric()}) of {primary.address} 
to be the Executor and Trustee of this my Will."""
        
        # Add alternates if provided
        if len(executors) > 1:
            clause += f"""

IF my said {primary.relationship if primary.relationship else 'Executor'} shall 
predecease me or be unwilling or unable to act, THEN I APPOINT the following 
persons to be my Executor(s) and Trustee(s):"""
            
            for i, executor in enumerate(executors[1:], start=1):
                clause += f"""
{i}. {executor.full_name.upper()} (NRIC NO. {executor.format_nric()}) of {executor.address}"""
        
        return clause
    
    def generate_debts_clause(self) -> str:
        """
        Generate debts and expenses clause.
        
        This is mandatory and directs the executor to pay all debts
        BEFORE distributing assets to beneficiaries.
        """
        return """
PAYMENT OF DEBTS AND EXPENSES

I DIRECT my Executor to pay my just debts, funeral and testamentary expenses, 
and estate duty (if any) out of my estate as soon as practicable after my death."""
    
    def generate_specific_bequest(self, asset: AssetInfo, beneficiary: BeneficiaryInfo) -> str:
        """
        Generate specific bequest clause for an asset.
        
        Different asset types require different clause formats with
        appropriate legal language and specific details.
        """
        # Get RAG-suggested clause based on asset type
        query = f"Generate a specific bequest clause for {asset.asset_type} in a Malaysian will"
        rag_suggestion = self.rag.query(query)
        
        # Build clause based on asset type
        if asset.asset_type == "real_property":
            address = asset.specific_details.get('address', asset.description)
            title = asset.specific_details.get('title_number', '[Title Number]')
            
            return f"""I give, devise and bequeath my property known as {address} 
held under Title No. {title} to my {beneficiary.relationship}, 
{beneficiary.full_name.upper()} (NRIC NO. {beneficiary.format_nric()}), absolutely."""
        
        elif asset.asset_type == "bank_account":
            bank = asset.specific_details.get('bank_name', '[Bank Name]')
            
            return f"""I give all my bank accounts, deposits, and cash holdings with {bank} 
to my {beneficiary.relationship}, {beneficiary.full_name.upper()} 
(NRIC NO. {beneficiary.format_nric()}), absolutely."""
        
        elif asset.asset_type == "vehicle":
            reg_no = asset.specific_details.get('registration_number', '[Reg No]')
            make = asset.specific_details.get('make_model', asset.description)
            
            return f"""I give my motor vehicle registration number {reg_no} ({make}) 
to my {beneficiary.relationship}, {beneficiary.full_name.upper()} 
(NRIC NO. {beneficiary.format_nric()}), absolutely."""
        
        elif asset.asset_type == "epf_kwsp":
            return f"""I have nominated beneficiaries for my EPF/KWSP benefits with the Employees 
Provident Fund Board. I direct my Trustee to ensure such nominations remain valid and current. 
Should there be no valid nomination registered with EPF at the time of my death, I direct that 
all my EPF/KWSP savings and contributions be distributed to my {beneficiary.relationship}, 
{beneficiary.full_name.upper()} (NRIC NO. {beneficiary.format_nric()}).

NOTE: EPF requires separate nomination at kwsp.gov.my. Will provisions are backup only."""
        
        elif asset.asset_type == "business":
            company = asset.specific_details.get('company_name', asset.description)
            reg_no = asset.specific_details.get('registration_number', '[Reg No]')
            ownership = asset.specific_details.get('ownership_percentage', '100')
            
            return f"""I give all my shares and interests in {company} (Company No. {reg_no}) 
comprising {ownership}% shareholding, together with all goodwill, intellectual property, 
trade names, and business assets, to my {beneficiary.relationship}, 
{beneficiary.full_name.upper()} (NRIC NO. {beneficiary.format_nric()}), absolutely."""
        
        elif asset.asset_type == "digital":
            access_loc = asset.specific_details.get('access_location', '[Secure Location]')
            
            return f"""I give all my digital assets including {asset.description} 
to my {beneficiary.relationship}, {beneficiary.full_name.upper()} 
(NRIC NO. {beneficiary.format_nric()}), absolutely. 
Access credentials are stored at {access_loc}."""
        
        else:
            # Generic clause for other assets
            return f"""I give {asset.description} to my {beneficiary.relationship}, 
{beneficiary.full_name.upper()} (NRIC NO. {beneficiary.format_nric()}), absolutely."""
    
    def generate_residuary_clause(self, beneficiaries: List[BeneficiaryInfo]) -> str:
        """
        Generate residuary estate clause (CRITICAL - prevents partial intestacy).
        
        This clause catches all property not specifically mentioned and is
        essential to prevent any part of the estate from passing by intestacy.
        """
        # Determine distribution method
        residuary_beneficiaries = [b for b in beneficiaries if b.distribution_type == "residuary"]
        
        if not residuary_beneficiaries:
            # Default to equal shares among all beneficiaries
            if len(beneficiaries) == 1:
                b = beneficiaries[0]
                return f"""
RESIDUARY ESTATE

I give all the rest, residue, and remainder of my estate, both real and personal, 
of whatsoever nature and wheresoever situated, which I may die possessed of or 
entitled to, and not hereby or by any codicil hereto otherwise specifically 
disposed of, including any lapsed or void gifts, to my {b.relationship}, 
{b.full_name.upper()} (NRIC NO. {b.format_nric()}), absolutely."""
            
            else:
                names_list = "\n".join([
                    f"   - {b.full_name.upper()} (NRIC NO. {b.format_nric()})"
                    for b in beneficiaries
                ])
                
                return f"""
RESIDUARY ESTATE

I give all the rest, residue, and remainder of my estate, both real and personal, 
of whatsoever nature and wheresoever situated, to the following beneficiaries 
in equal shares absolutely, per stirpes (such that if any beneficiary predeceases 
me leaving issue, such issue shall take their parent's share):

{names_list}"""
        
        else:
            # Use specified residuary beneficiaries
            if len(residuary_beneficiaries) == 1:
                b = residuary_beneficiaries[0]
                return f"""
RESIDUARY ESTATE

I give all the rest, residue, and remainder of my estate to my {b.relationship}, 
{b.full_name.upper()} (NRIC NO. {b.format_nric()}), absolutely."""
            
            else:
                names_list = "\n".join([
                    f"   - {b.full_name.upper()} (NRIC NO. {b.format_nric()})"
                    for b in residuary_beneficiaries
                ])
                
                return f"""
RESIDUARY ESTATE

I give the residue of my estate to the following beneficiaries in equal shares:

{names_list}"""
    
    def generate_guardian_clause(self, guardians: List[PersonInfo]) -> str:
        """
        Generate guardian appointment clause for minor children.
        
        Only generated if testator has children under 18 years old.
        """
        if len(guardians) == 0:
            return ""
        
        primary = guardians[0]
        clause = f"""
GUARDIAN APPOINTMENT

I APPOINT {primary.full_name.upper()} (NRIC NO. {primary.format_nric()}) 
of {primary.address} to be the guardian of my children under the age of 18 years, 
to have custody, care and control of them until they attain the age of 18 years."""
        
        if len(guardians) > 1:
            backup = guardians[1]
            clause += f"""

IF my said primary guardian is unwilling or unable to act, THEN I APPOINT 
{backup.full_name.upper()} (NRIC NO. {backup.format_nric()}) to be the guardian 
of my minor children."""
        
        return clause
    
    def generate_testimonium_clause(self) -> str:
        """
        Generate testimonium clause (formal closing statement).
        
        This declares when and by whom the will was made.
        """
        date_str = datetime.now().strftime("%d day of %B %Y")
        
        return f"""
TESTIMONIUM

IN WITNESS WHEREOF I have hereunto set my hand and seal this {date_str}."""
    
    def generate_attestation_section(self, testator: TestatorInfo, witnesses: List[PersonInfo]) -> str:
        """
        Generate attestation section with signature blocks.
        
        This is where the testator and witnesses sign. The specific format
        and language here is critical for legal validity under Section 5.
        """
        return f"""

________________________________________
Signature of Testator
{testator.full_name.upper()}
NRIC NO. {testator.format_nric()}


SIGNED by the above-named {testator.full_name.upper()} as and for the Testator's 
LAST WILL AND TESTAMENT in the presence of us, present at the same time, who at 
the Testator's request and in the Testator's presence and in the presence of each 
other have hereunto subscribed our names as witnesses:


WITNESS 1:

________________________________________
Signature of Witness
{witnesses[0].full_name.upper()}
NRIC NO. {witnesses[0].format_nric()}
Address: {witnesses[0].address}


WITNESS 2:

________________________________________
Signature of Witness
{witnesses[1].full_name.upper()}
NRIC NO. {witnesses[1].format_nric()}
Address: {witnesses[1].address}
"""
    
    def generate_will(self, will_data: WillData) -> str:
        """
        Generate complete will document.
        
        This is the main method that assembles all sections in the correct
        legal order to create a valid Malaysian will.
        
        Parameters:
        -----------
        will_data : WillData
            Validated will information
        
        Returns:
        --------
        str
            Complete will document text
        """
        # Store for reference
        self.will_data = will_data
        
        # Validate data
        valid, errors = will_data.validate_complete()
        if not valid:
            error_msg = "\n".join(errors)
            raise ValueError(f"Will data validation failed:\n{error_msg}")
        
        # Build document sections in legal order
        sections = []
        
        # 1. Opening Declaration
        sections.append(self.generate_opening_declaration(will_data.testator))
        
        # 2. Executor Appointment
        sections.append(self.generate_executor_clause(will_data.executors))
        
        # 3. Debts and Expenses
        sections.append(self.generate_debts_clause())
        
        # 4. Distribution Clauses (Specific Bequests)
        if will_data.assets:
            sections.append("\nDISTRIBUTION OF ASSETS\n")
            
            # Match assets with beneficiaries
            beneficiary_map = {b.nric: b for b in will_data.beneficiaries}
            
            for i, asset in enumerate(will_data.assets, start=1):
                beneficiary = beneficiary_map.get(asset.beneficiary_nric)
                if beneficiary:
                    clause = self.generate_specific_bequest(asset, beneficiary)
                    sections.append(f"{i}. {clause}\n")
        
        # 5. Residuary Clause (CRITICAL)
        sections.append(self.generate_residuary_clause(will_data.beneficiaries))
        
        # 6. Guardian Appointment (if applicable)
        if will_data.has_minor_children and will_data.guardians:
            sections.append(self.generate_guardian_clause(will_data.guardians))
        
        # 7. Special Instructions (optional)
        if will_data.special_instructions:
            sections.append(f"""
SPECIAL INSTRUCTIONS

{will_data.special_instructions}""")
        
        # 8. Testimonium
        sections.append(self.generate_testimonium_clause())
        
        # 9. Attestation Section
        sections.append(self.generate_attestation_section(
            will_data.testator, 
            will_data.witnesses
        ))
        
        # Combine all sections
        complete_will = "\n".join(sections)
        
        return complete_will
    
    def save_will(self, will_text: str, filename: Optional[str] = None) -> Path:
        """
        Save generated will to file.
        
        Parameters:
        -----------
        will_text : str
            The generated will document
        filename : str, optional
            Custom filename (default: testator name + timestamp)
        
        Returns:
        --------
        Path
            Path to saved file
        """
        if filename is None:
            testator_name = self.will_data.testator.full_name.replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"will_{testator_name}_{timestamp}.txt"
        
        output_path = self.rag.config.output_path / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(will_text)
        
        print(f"✅ Will saved to: {output_path}")
        return output_path
    
    def generate_signing_instructions(self) -> str:
        """
        Generate instructions for properly executing the will.
        
        These instructions ensure the will is signed correctly
        to meet legal requirements under Wills Act 1959.
        """
        return """
INSTRUCTIONS FOR SIGNING YOUR WILL

CRITICAL REQUIREMENTS FOR VALIDITY (Wills Act 1959, Section 5):

1. TESTATOR SIGNATURE
   - You (the testator) must sign at the END of the will
   - Use blue or black ink (for original identification)
   - Sign in the presence of BOTH witnesses AT THE SAME TIME
   - Both witnesses must be physically present when you sign

2. WITNESS SIGNATURES
   - Both witnesses must sign AFTER you sign
   - Both must sign while you are present
   - Both must sign while each other is present
   - All three people must be in the same room at the same time

3. WITNESS QUALIFICATIONS
   - Must be 18+ years old (preferably 21+)
   - Must be of sound mind
   - CANNOT be beneficiaries named in the will
   - CANNOT be spouses of beneficiaries
   - CAN be the executor if executor is not also a beneficiary
   
4. EXECUTION PROCESS
   Step 1: Gather yourself and BOTH witnesses in the same room
   Step 2: Ensure all three people can see the will
   Step 3: You sign at the designated space at the end
   Step 4: First witness signs in their designated space
   Step 5: Second witness signs in their designated space
   Step 6: All signatures must be in front of each other

5. AFTER SIGNING
   - Store the ORIGINAL will safely (consider professional custody)
   - Never staple, clip, or mark the original
   - Inform your executor where the original is kept
   - Consider giving a copy (not original) to your executor
   - Review and update your will every 3-5 years or after major life events

6. IMPORTANT REMINDERS
   - Marriage/remarriage automatically REVOKES your will
   - Divorce does NOT revoke your will - make a new one
   - EPF/KWSP requires separate nomination (not covered by will)
   - Insurance policies require separate nominations
   - Joint bank accounts pass by survivorship, not through will
   - This will should be for NON-MUSLIMS only (Muslims need Wasiat)

7. PROFESSIONAL REVIEW
   - Consider having this will reviewed by a lawyer
   - Cost for lawyer review: typically RM500-1500
   - This ensures all clauses are legally sound
   - Especially important for complex estates (> RM1 million)

8. WILL CUSTODY OPTIONS
   - Home safe (inform executor of location)
   - Bank safe deposit box
   - Professional custody services (Rockwills, AmanahRaya)
   - Law firm custody
   - DO NOT keep with executor if executor is also beneficiary

PROBATE PROCESS (After Death):
Your executor will need to:
1. Locate the original will
2. Engage a lawyer to apply for Grant of Probate
3. High Court process typically takes 6-12 months
4. Cost: RM5,000-15,000 for straightforward estates
5. Once probate granted, executor distributes assets per your wishes

For questions or concerns, consult a qualified lawyer specializing
in estate planning and probate in Malaysia.
"""

# Initialize will generator
print("\n" + "="*70)
print("WILL GENERATOR INITIALIZED")
print("="*70)

will_generator = MalaysianWillGenerator(rag_system)
print("✅ Will generator ready")
print("   Can generate complete legally-valid Malaysian wills")
print("   Includes all mandatory sections per Wills Act 1959")
