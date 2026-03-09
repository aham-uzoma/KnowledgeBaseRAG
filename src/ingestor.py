import os
from pathlib import Path
from loguru import logger

# New imports for speed optimization
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

class KnowledgeIngestor:
    def __init__(self):
        # 1. NEW: Explicitly configure Accelerator to use CPU
        # This prevents the 'meta tensor' error by avoiding device guesswork
        accelerator_options = AcceleratorOptions(
            device="cpu",      # Force CPU to avoid meta tensor issues
            num_threads=4      # Use 4 threads for faster processing without OCR
        )
        # 1. Setup Pipeline Options (Disabling OCR for speed)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # <--- The "Speed Booster"
        pipeline_options.do_table_structure = True
        pipeline_options.accelerator_options = accelerator_options # Apply the fix 

        # 2. Initialize the converter with a faster backend
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend # <--- Faster for digital PDFs
                )
            }
        )
        
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_pdf(self, file_name: str):
        """
        Converts PDF to Markdown.
        By default, it uses fast digital extraction (No OCR).
        """
        input_path = self.raw_dir / file_name
        output_path = self.processed_dir / f"{input_path.stem}.md"

        logger.info(f"🚀 Starting FAST conversion for: {file_name}")

        try:
            # 1. Convert the PDF
            result = self.converter.convert(input_path)
            
            # 2. Export it to Markdown
            markdown_content = result.document.export_to_markdown()
            
            # Check if we got any text (If it's an image-only PDF, content will be empty)
            if not markdown_content.strip():
                logger.warning(f"⚠️ {file_name} appears to be a scan. No text found without OCR.")

            # 3. Save to processed folder
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
                
            logger.success(f"Success! Clean data saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to process {file_name}: {e}")
            return None

if __name__ == "__main__":
    ingestor = KnowledgeIngestor()
    pdf_files = list(Path("data/raw").glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDFs found in data/raw!")
    else:
        for pdf in pdf_files:
            ingestor.process_pdf(pdf.name)



# import os
# from pathlib import Path
# from docling.document_converter import DocumentConverter
# from loguru import logger

# class KnowledgeIngestor:
#     def __init__(self):
#         # Initialize the converter - this is the "brain" of the parsing
#         self.converter = DocumentConverter()
        
#         # Define our folder paths using the structure we built
#         self.raw_dir = Path("data/raw")
#         self.processed_dir = Path("data/processed")
        
#         # Create directories if they don't exist
#         self.processed_dir.mkdir(parents=True, exist_ok=True)

#     def process_pdf(self, file_name: str):
#         input_path = self.raw_dir / file_name
#         output_path = self.processed_dir / f"{input_path.stem}.md"

#         logger.info(f"🚀 Starting conversion for: {file_name}")

#         try:
#             # 1. Convert the PDF to a structured Docling document
#             result = self.converter.convert(input_path)
            
#             # 2. Export it to Markdown (best format for AI/LLMs)
#             markdown_content = result.document.export_to_markdown()
            
#             # 3. Save the clean text to our processed folder
#             with open(output_path, "w", encoding="utf-8") as f:
#                 f.write(markdown_content)
                
#             logger.success(f"Success! Clean data saved to: {output_path}")
#             return output_path

#         except Exception as e:
#             logger.error(f"Failed to process {file_name}: {e}")
#             return None

# if __name__ == "__main__":
#     # This part runs when you execute the script directly
#     ingestor = KnowledgeIngestor()
    
#     # List all PDFs in the raw folder
#     pdf_files = list(Path("data/raw").glob("*.pdf"))
    
#     if not pdf_files:
#         logger.warning("No PDFs found in data/raw! Drop a file there and try again.")
#     else:
#         for pdf in pdf_files:
#             ingestor.process_pdf(pdf.name)