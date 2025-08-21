"""
Financial Named Entity Recognition (NER) System
================================================
A flexible system for extracting financial entities from various document types
including chats, emails, PDFs, and other unstructured text formats.

GPU Setup Instructions:
----------------------
For GPU support, install PyTorch with CUDA:

# For CUDA 11.8 (most common):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (latest):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU availability:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

Required packages:
pip install transformers torch spacy pandas openpyxl
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

# Install required packages:
# pip install transformers torch spacy pandas openpyxl

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    BertForTokenClassification,
    BertTokenizer
)
import spacy
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinancialEntityType(Enum):
    """Enumeration of financial entity types"""
    COUNTERPARTY = "COUNTERPARTY"
    NOTIONAL = "NOTIONAL"
    AMOUNT = "AMOUNT"
    CURRENCY = "CURRENCY"
    ISIN = "ISIN"
    UNDERLYING = "UNDERLYING"
    MATURITY = "MATURITY"
    BID = "BID"
    OFFER = "OFFER"
    BID_ASK = "BID_ASK"
    RATE = "RATE"
    PAYMENT_FREQUENCY = "PAYMENT_FREQUENCY"
    DATE = "DATE"
    PRODUCT_TYPE = "PRODUCT_TYPE"
    REFERENCE_RATE = "REFERENCE_RATE"


@dataclass
class FinancialEntity:
    """Data class for storing extracted financial entities"""
    text: str
    entity_type: FinancialEntityType
    start_pos: int
    end_pos: int
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FinancialNERExtractor:
    """
    Main class for financial entity extraction using various NER models
    """
    
    def __init__(self, model_name: str = "dslim/bert-base-NER", use_gpu: bool = True):
        """
        Initialize the NER extractor with a specified model
        
        Args:
            model_name: HuggingFace model name or path to local model
            use_gpu: Whether to use GPU for inference (will fallback to CPU if GPU not available)
        """
        self.model_name = model_name
        
        # GPU Detection and Setup
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"GPU detected! Using: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.device = torch.device("cpu")
            if use_gpu and not torch.cuda.is_available():
                logger.warning("GPU requested but not available. Falling back to CPU.")
                logger.info("To use GPU, ensure you have CUDA-compatible PyTorch installed:")
                logger.info("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            else:
                logger.info("Using CPU for inference")
        
        self.tokenizer = None
        self.model = None
        self.nlp_pipeline = None
        self.custom_patterns = self._initialize_patterns()
        
        logger.info(f"Initializing NER extractor with model: {model_name}")
        logger.info(f"Device selected: {self.device}")
        
        self._load_model()
        
    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for financial entities"""
        return {
            'ISIN': re.compile(r'\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b'),
            'NOTIONAL': re.compile(r'\b\d+(?:\.\d+)?(?:\s*(?:mio|million|mn|bn|billion|k|thousand|m))\b', re.IGNORECASE),
            'CURRENCY': re.compile(r'\b(USD|EUR|GBP|JPY|CHF|AUD|CAD|CNY|SEK|NZD)\b'),
            'BID': re.compile(r'\b(?:bid|offer)\s+(?:estr|libor|sofr|euribor|sonia)[+\-]?\d+(?:\.\d+)?(?:bps|bp)?\b', re.IGNORECASE),
            'RATE': re.compile(r'\b(?:estr|libor|sofr|euribor|sonia)[+\-]?\d+(?:\.\d+)?(?:bps|bp)?\b', re.IGNORECASE),
            'MATURITY': re.compile(r'\b\d+[YMWDymwd](?:\s+[A-Z]+)?\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b'),
            'PAYMENT_FREQUENCY': re.compile(r'\b(?:quarterly|monthly|annually|semi-annually|daily|weekly)\b', re.IGNORECASE),
            'UNDERLYING': re.compile(r'\b[A-Z]{3,}\s+(?:FLOAT|FIXED|SWAP|BOND|NOTE)\b'),
            'DATE': re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),
        }
    
    def _load_model(self):
        """Load the pre-trained NER model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easy inference
            self.nlp_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                aggregation_strategy="simple"
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def extract_entities_with_model(self, text: str) -> List[FinancialEntity]:
        """
        Extract entities using the loaded NER model
        
        Args:
            text: Input text to process
            
        Returns:
            List of extracted financial entities
        """
        entities = []
        
        try:
            # Run NER pipeline
            model_outputs = self.nlp_pipeline(text)
            
            for output in model_outputs:
                # Filter out low confidence entities and numbers that are likely timestamps
                if output['score'] < 0.7:
                    continue
                    
                # Skip single/double digit numbers (likely timestamps or dates)
                if output['word'].strip().isdigit() and len(output['word'].strip()) <= 2:
                    continue
                    
                # Skip fragments (tokens starting with ##)
                if output['word'].startswith('##'):
                    continue
                
                # Map general entities to financial entities
                entity_type = self._map_to_financial_entity(output['entity_group'], output['word'])
                
                if entity_type:
                    entity = FinancialEntity(
                        text=output['word'],
                        entity_type=entity_type,
                        start_pos=output['start'],
                        end_pos=output['end'],
                        confidence=output['score']
                    )
                    entities.append(entity)
                    
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            
        return entities
    
    def extract_entities_with_patterns(self, text: str) -> List[FinancialEntity]:
        """
        Extract entities using regex patterns
        
        Args:
            text: Input text to process
            
        Returns:
            List of extracted financial entities
        """
        entities = []
        
        for entity_type, pattern in self.custom_patterns.items():
            matches = pattern.finditer(text)
            
            for match in matches:
                # Safely get the entity type from enum
                try:
                    fin_entity_type = FinancialEntityType[entity_type]
                except KeyError:
                    logger.warning(f"Unknown entity type in patterns: {entity_type}")
                    continue
                    
                entity = FinancialEntity(
                    text=match.group(),
                    entity_type=fin_entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=1.0  # High confidence for pattern matches
                )
                entities.append(entity)
                
        return entities
    
    def extract_all_entities(self, text: str, use_patterns: bool = True) -> List[FinancialEntity]:
        """
        Extract entities using both model and patterns
        
        Args:
            text: Input text to process
            use_patterns: Whether to use regex patterns in addition to model
            
        Returns:
            Combined list of extracted entities
        """
        entities = self.extract_entities_with_model(text)
        
        if use_patterns:
            pattern_entities = self.extract_entities_with_patterns(text)
            entities.extend(pattern_entities)
            
        # Remove duplicates and merge overlapping entities
        entities = self._merge_entities(entities)
        
        return sorted(entities, key=lambda x: x.start_pos)
    
    def _map_to_financial_entity(self, entity_label: str, text: str = "") -> Optional[FinancialEntityType]:
        """
        Map general NER labels to financial entity types with context awareness
        
        Args:
            entity_label: Label from the NER model
            text: The actual text of the entity for context
            
        Returns:
            Corresponding financial entity type or None
        """
        # Check if it's an amount with unit (mio, bn, etc.)
        if any(unit in text.lower() for unit in ['mio', 'million', 'bn', 'billion']):
            return FinancialEntityType.NOTIONAL
        
        # Basic mapping
        mapping = {
            'ORG': FinancialEntityType.COUNTERPARTY,
            'MONEY': FinancialEntityType.NOTIONAL,
            'DATE': FinancialEntityType.DATE,
            'PERCENT': FinancialEntityType.RATE,
        }
        
        # Don't map CARDINAL (numbers) unless they have financial context
        if entity_label.upper() == 'CARDINAL':
            # Only return AMOUNT if it's a large number or has units
            if len(text) > 2 or any(unit in text.lower() for unit in ['mio', 'million', 'bn', 'billion', 'k']):
                return FinancialEntityType.NOTIONAL
            return None
        
        return mapping.get(entity_label.upper())
    
    def _merge_entities(self, entities: List[FinancialEntity]) -> List[FinancialEntity]:
        """
        Merge overlapping entities, keeping the one with higher confidence
        
        Args:
            entities: List of entities to merge
            
        Returns:
            Merged list of entities
        """
        if not entities:
            return entities
            
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: (x.start_pos, -x.confidence))
        merged = []
        
        for entity in sorted_entities:
            # Check if entity overlaps with any already merged entity
            overlaps = False
            for merged_entity in merged:
                if (entity.start_pos < merged_entity.end_pos and 
                    entity.end_pos > merged_entity.start_pos):
                    overlaps = True
                    break
                    
            if not overlaps:
                merged.append(entity)
                
        return merged
    
    def extract_financial_entities_structured(self, text: str) -> Dict[str, str]:
        """
        Extract financial entities and return in a structured format
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with entity types as keys and extracted values
        """
        # Structure the results
        structured = {}
        
        # Custom extraction for specific patterns
        # Counterparty - look for BANK or institution names
        counterparty_match = re.search(r'\b(BANK\s+[A-Z]+|[A-Z]{3,}\s+(?:Bank|Corp|Ltd|Inc))\b', text)
        if counterparty_match:
            structured['Counterparty'] = counterparty_match.group().strip()
        
        # Notional - amounts with units
        notional_match = re.search(r'\b(\d+(?:\.\d+)?)\s*(mio|million|mn|bn|billion|k|m)\b', text, re.IGNORECASE)
        if notional_match:
            structured['Notional'] = f"{notional_match.group(1)} {notional_match.group(2)}"
        
        # ISIN
        isin_match = re.search(r'\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b', text)
        if isin_match:
            structured['ISIN'] = isin_match.group()
        
        # Underlying - product description
        underlying_match = re.search(r'\b([A-Z]{3,}\s+(?:FLOAT|FIXED|SWAP|BOND|NOTE)(?:\s+\d{1,2}/\d{1,2}/\d{2,4})?)\b', text)
        if underlying_match:
            structured['Underlying'] = underlying_match.group().strip()
        
        # Maturity - look for patterns like "2Y EVG"
        maturity_match = re.search(r'\b(\d+[YMWDymwd])(?:\s+[A-Z]+)?\b', text)
        if maturity_match:
            # Check if there's text after the tenor
            full_match = re.search(r'\b(\d+[YMWDymwd](?:\s+[A-Z]+)?)\b', text)
            if full_match:
                structured['Maturity'] = full_match.group().strip()
        
        # Bid/Offer with rates
        # Look for rate patterns after "offer" keyword
        offer_line_match = re.search(r'offer.*?(estr[+\-]?\d+(?:bps|bp)?)', text, re.IGNORECASE | re.DOTALL)
        if offer_line_match:
            structured['Bid'] = offer_line_match.group(1)
        else:
            # Just look for rate pattern
            rate_match = re.search(r'((?:estr|libor|sofr|euribor|sonia)[+\-]?\d+(?:bps|bp)?)', text, re.IGNORECASE)
            if rate_match:
                structured['Bid'] = rate_match.group()
        
        # Payment Frequency
        freq_match = re.search(r'\b(quarterly|monthly|annually|semi-annually|daily|weekly)\b', text, re.IGNORECASE)
        if freq_match:
            structured['PaymentFrequency'] = freq_match.group().capitalize()
        
        return structured
    
    def process_document(self, 
                         document_path: str, 
                         document_type: str = "text") -> Dict[str, Any]:
        """
        Process a document and extract financial entities
        
        Args:
            document_path: Path to the document
            document_type: Type of document (text, chat, email, pdf)
            
        Returns:
            Dictionary containing extracted entities and metadata
        """
        logger.info(f"Processing document: {document_path}")
        
        # Load document based on type
        text = self._load_document(document_path, document_type)
        
        # Extract entities
        entities = self.extract_all_entities(text)
        
        # Structure the output
        result = {
            'document_path': document_path,
            'document_type': document_type,
            'processed_at': datetime.now().isoformat(),
            'entities': self._entities_to_dict(entities),
            'statistics': self._calculate_statistics(entities),
            'raw_text': text[:500] + '...' if len(text) > 500 else text
        }
        
        return result
    
    def extract_financial_entities_structured(self, text: str) -> Dict[str, str]:
        """
        Extract financial entities and return in a structured format
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with entity types as keys and extracted values
        """
        # Structure the results
        structured = {}
        
        # Custom extraction for specific patterns
        # Counterparty - look for BANK or institution names
        counterparty_match = re.search(r'\b(BANK\s+[A-Z]+|[A-Z]{3,}\s+(?:Bank|Corp|Ltd|Inc))\b', text)
        if counterparty_match:
            structured['Counterparty'] = counterparty_match.group().strip()
        
        # Notional - amounts with units
        notional_match = re.search(r'\b(\d+(?:\.\d+)?)\s*(mio|million|mn|bn|billion|k|m)\b', text, re.IGNORECASE)
        if notional_match:
            structured['Notional'] = f"{notional_match.group(1)} {notional_match.group(2)}"
        
        # ISIN
        isin_match = re.search(r'\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b', text)
        if isin_match:
            structured['ISIN'] = isin_match.group()
        
        # Underlying - product description
        underlying_match = re.search(r'\b([A-Z]{3,}\s+(?:FLOAT|FIXED|SWAP|BOND|NOTE)(?:\s+\d{1,2}/\d{1,2}/\d{2,4})?)\b', text)
        if underlying_match:
            structured['Underlying'] = underlying_match.group().strip()
        
        # Maturity - look for patterns like "2Y EVG"
        maturity_match = re.search(r'\b(\d+[YMWDymwd])(?:\s+[A-Z]+)?\b', text)
        if maturity_match:
            # Check if there's text after the tenor
            full_match = re.search(r'\b(\d+[YMWDymwd](?:\s+[A-Z]+)?)\b', text)
            if full_match:
                structured['Maturity'] = full_match.group().strip()
        
        # Bid/Offer with rates
        # Look for rate patterns after "offer" keyword
        offer_line_match = re.search(r'offer.*?(estr[+\-]?\d+(?:bps|bp)?)', text, re.IGNORECASE | re.DOTALL)
        if offer_line_match:
            structured['Bid'] = offer_line_match.group(1)
        else:
            # Just look for rate pattern
            rate_match = re.search(r'((?:estr|libor|sofr|euribor|sonia)[+\-]?\d+(?:bps|bp)?)', text, re.IGNORECASE)
            if rate_match:
                structured['Bid'] = rate_match.group()
        
        # Payment Frequency
        freq_match = re.search(r'\b(quarterly|monthly|annually|semi-annually|daily|weekly)\b', text, re.IGNORECASE)
        if freq_match:
            structured['PaymentFrequency'] = freq_match.group().capitalize()
        
        return structured
    
    def _load_document(self, path: str, doc_type: str) -> str:
        """
        Load document content based on type
        
        Args:
            path: Path to document
            doc_type: Type of document
            
        Returns:
            Text content of the document
        """
        # This is a simplified loader - extend based on your needs
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return ""
    
    def _entities_to_dict(self, entities: List[FinancialEntity]) -> List[Dict]:
        """Convert entities to dictionary format"""
        return [
            {
                'text': e.text,
                'type': e.entity_type.value,
                'start': e.start_pos,
                'end': e.end_pos,
                'confidence': e.confidence,
                'metadata': e.metadata
            }
            for e in entities
        ]
    
    def _calculate_statistics(self, entities: List[FinancialEntity]) -> Dict:
        """Calculate statistics about extracted entities"""
        stats = {}
        
        # Count by type
        type_counts = {}
        for entity in entities:
            type_name = entity.entity_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
        stats['entity_counts'] = type_counts
        stats['total_entities'] = len(entities)
        stats['avg_confidence'] = (
            sum(e.confidence for e in entities) / len(entities) 
            if entities else 0
        )
        
        return stats
    
    def export_results(self, results: Dict, output_format: str = "json") -> str:
        """
        Export extraction results in various formats
        
        Args:
            results: Extraction results dictionary
            output_format: Format for export (json, csv, excel)
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "json":
            output_path = f"extraction_results_{timestamp}.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
        elif output_format == "csv":
            output_path = f"extraction_results_{timestamp}.csv"
            df = pd.DataFrame(results['entities'])
            df.to_csv(output_path, index=False)
            
        elif output_format == "excel":
            output_path = f"extraction_results_{timestamp}.xlsx"
            df = pd.DataFrame(results['entities'])
            df.to_excel(output_path, index=False)
            
        logger.info(f"Results exported to: {output_path}")
        return output_path


class FinancialNERTrainer:
    """
    Class for fine-tuning NER models on financial data
    """
    
    def __init__(self, base_model: str = "bert-base-uncased", use_gpu: bool = True):
        """
        Initialize the trainer
        
        Args:
            base_model: Base model to fine-tune
            use_gpu: Whether to use GPU for training
        """
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(base_model)
        self.model = None
        
        if self.device.type == "cuda":
            logger.info(f"Trainer using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Trainer using CPU")
        
    def prepare_training_data(self, 
                             annotations_file: str) -> Tuple[List, List]:
        """
        Prepare training data from annotated file
        
        Args:
            annotations_file: Path to file with annotated data
            
        Returns:
            Tuple of training features and labels
        """
        # Implementation for loading and preparing training data
        # This would load your annotated financial documents
        pass
    
    def train(self, 
              train_data: List, 
              val_data: List, 
              epochs: int = 3, 
              batch_size: int = 16):
        """
        Fine-tune the model on financial data
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Implementation for model training
        # This would include the training loop, optimization, etc.
        pass
    
    def save_model(self, output_dir: str):
        """Save the fine-tuned model"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to: {output_dir}")


# Example usage
def check_gpu_setup():
    """
    Utility function to check GPU setup and provide installation instructions
    """
    print("\n" + "="*60)
    print("GPU SETUP CHECK")
    print("="*60)
    
    print(f"\nPyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    - Compute Capability: {props.major}.{props.minor}")
            
        # Test GPU
        print("\n  Testing GPU...")
        try:
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.mm(test_tensor, test_tensor)
            print("  ✓ GPU computation test passed!")
        except Exception as e:
            print(f"  ✗ GPU test failed: {e}")
            
    else:
        print("✗ CUDA is NOT available")
        print("\nTo enable GPU support:")
        print("1. Check if you have an NVIDIA GPU")
        print("2. Install CUDA-enabled PyTorch:")
        print("\n   # For Windows/Linux with CUDA 11.8:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n   # For Windows/Linux with CUDA 12.1:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\n3. Verify installation:")
        print("   python -c \"import torch; print(torch.cuda.is_available())\"")
        
        # Check for common issues
        print("\nCommon issues:")
        print("- Ensure NVIDIA drivers are installed and up to date")
        print("- Check if your GPU is CUDA-compatible (compute capability >= 3.5)")
        print("- Make sure no other process is using all GPU memory")
    
    print("\n" + "="*60 + "\n")
    return torch.cuda.is_available()


# Example usage
def main():
    """
    Example usage of the Financial NER system
    """
    
    # First, check GPU setup
    gpu_available = check_gpu_setup()
    
    # Initialize the extractor (will automatically use GPU if available)
    extractor = FinancialNERExtractor(
        model_name="dslim/bert-base-NER",  # You can use other models
        use_gpu=True  # Will use GPU if available, otherwise CPU
    )
    
    # Sample financial text (like from your chat example)
    sample_text = """
    11:49:05 I'll revert regarding BANK ABC to try to do another 200 mio at 2Y
    FR001400QV82 AVMAFC FLOAT 06/30/28
    offer 2Y EVG estr+45bps
    estr average Estr average / Quarterly interest payment
    """
    
    print("\n=== STRUCTURED FINANCIAL ENTITY EXTRACTION ===\n")
    
    # Extract structured entities
    structured_entities = extractor.extract_financial_entities_structured(sample_text)
    
    # Display in desired format
    print("Extracted Entities:")
    print("-" * 40)
    
    # Ordered display
    display_order = [
        ('Counterparty', structured_entities.get('Counterparty')),
        ('Notional', structured_entities.get('Notional')),
        ('ISIN', structured_entities.get('ISIN')),
        ('Underlying', structured_entities.get('Underlying')),
        ('Maturity', structured_entities.get('Maturity')),
        ('Bid', structured_entities.get('Bid')),
        ('Offer', structured_entities.get('Offer')),
        ('PaymentFrequency', structured_entities.get('PaymentFrequency'))
    ]
    
    for key, value in display_order:
        if value:
            print(f"{key} ► {value}")
        elif key == 'Offer':
            print(f"{key}")
    
    print("\n" + "=" * 40)
    
    # Also show detailed extraction if needed
    show_detailed = input("\nShow detailed entity extraction? (y/n): ").lower() == 'y'
    
    if show_detailed:
        print("\n=== DETAILED ENTITY EXTRACTION ===\n")
        entities = extractor.extract_all_entities(sample_text)
        
        for entity in entities:
            print(f"Text: '{entity.text}'")
            print(f"Type: {entity.entity_type.value}")
            print(f"Position: [{entity.start_pos}:{entity.end_pos}]")
            print(f"Confidence: {entity.confidence:.2f}")
            print("-" * 40)
    
    # Show performance info if using GPU
    if torch.cuda.is_available() and extractor.device.type == "cuda":
        print(f"\n=== GPU Performance ===")
        print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    

if __name__ == "__main__":
    main()