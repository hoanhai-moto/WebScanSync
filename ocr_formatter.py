import json
from typing import Dict, Any, List, Optional

def format_ocr_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format raw OCR data into a structured representation.
    
    Args:
        raw_data: The raw OCR data from Document Intelligence API or saved JSON
        
    Returns:
        Formatted data with organized text content and metadata
    """
    if not raw_data or raw_data.get("status") != "completed":
        return {"status": "error", "message": "Invalid or failed OCR data"}
    
    # Initialize formatted structure
    formatted_data = {
        "document_id": raw_data.get("document_id", ""),
        "text_content": raw_data.get("raw_text", ""),
        "pages": []
    }
    
    # Process each page
    for page in raw_data.get("pages", []):
        page_data = {
            "page_number": page.get("page_number", 0),
            "dimensions": {
                "width": page.get("width", 0),
                "height": page.get("height", 0),
                "unit": page.get("unit", "")
            },
            "paragraphs": [],
            "handwritten_sections": [],
            "table_data": []
        }
        
        # Group lines into paragraphs (simple approach: consecutive lines)
        current_paragraph = []
        last_y_position = None
        y_threshold = 10  # Adjust based on document characteristics
        
        for line in sorted(page.get("lines", []), key=lambda x: x.get("bounding_box", "")):
            # Extract y-position from bounding box (assuming format is "[x1, y1], [x2, y2], [x3, y3], [x4, y4]")
            try:
                bbox = line.get("bounding_box", "")
                if bbox and "," in bbox:
                    y_position = float(bbox.split(",")[1].split("]")[0].strip())
                    
                    # Check if this is a new paragraph based on vertical spacing
                    if last_y_position is not None and abs(y_position - last_y_position) > y_threshold:
                        if current_paragraph:
                            page_data["paragraphs"].append({
                                "text": " ".join([l.get("text", "") for l in current_paragraph]),
                                "confidence": sum(l.get("confidence", 0) for l in current_paragraph) / len(current_paragraph) if current_paragraph else 0,
                                "lines": current_paragraph
                            })
                            current_paragraph = []
                    
                    current_paragraph.append(line)
                    last_y_position = y_position
                else:
                    current_paragraph.append(line)
            except Exception:
                current_paragraph.append(line)
        
        # Add the last paragraph
        if current_paragraph:
            page_data["paragraphs"].append({
                "text": " ".join([l.get("text", "") for l in current_paragraph]),
                "confidence": sum(l.get("confidence", 0) for l in current_paragraph) / len(current_paragraph) if current_paragraph else 0,
                "lines": current_paragraph
            })
        
        # Extract handwritten sections
        handwritten_lines = [line for line in page.get("lines", []) if line.get("is_handwritten", False)]
        if handwritten_lines:
            page_data["handwritten_sections"] = [{
                "text": line.get("text", ""),
                "confidence": line.get("confidence", 0),
                "bounding_box": line.get("bounding_box", "")
            } for line in handwritten_lines]
        
        # Add page data to formatted output
        formatted_data["pages"].append(page_data)
    
    # Add metadata
    formatted_data["metadata"] = {
        "total_pages": len(raw_data.get("pages", [])),
        "contains_handwritten_text": any(
            line.get("is_handwritten", False) 
            for page in raw_data.get("pages", []) 
            for line in page.get("lines", [])
        ),
        "confidence_score": _calculate_overall_confidence(raw_data)
    }
    
    return formatted_data

def _calculate_overall_confidence(raw_data: Dict[str, Any]) -> float:
    """Calculate the overall confidence score for the document."""
    confidences = []
    for page in raw_data.get("pages", []):
        for line in page.get("lines", []):
            if "confidence" in line and line["confidence"] is not None:
                confidences.append(line["confidence"])
    
    return sum(confidences) / len(confidences) if confidences else 0.0

def save_formatted_data(formatted_data: Dict[str, Any], output_path: str) -> None:
    """Save formatted OCR data to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)

def load_and_format_ocr_file(file_path: str) -> Dict[str, Any]:
    """Load OCR data from a file and format it."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return format_ocr_data(raw_data) 