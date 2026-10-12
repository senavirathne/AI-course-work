CONTENT
This dataset contains a collection of images designed to test, develop, and benchmark raw fabric defect detection algorithms. It provides diverse fabric samples sourced from several different manufacturers to ensure variability in texture and weaving. The dataset is specifically curated to support robust research in industrial anomaly detection and semantic segmentation within the textile industry.

FORMAT
Images: PNG grayscale (8-bit).
Annotations: COCO-compatible JSON format.
Metadata: CSV (Comma-Separated Values) format.

DATA ORGANIZATION
The dataset is organized into three main directories and includes four auxiliary annotation/metadata files in the root directory:

1. images directory (709 images)
Contains the high-resolution original images. To ensure data quality, the raw sensor captures were cropped (starting at coordinate X=120, with a width of 1792 pixels) to remove non-fabric dark borders and uneven peripheral illumination. All images have a resolution of 1792×1024 pixels with a single grayscale channel. This folder includes both defect-free and defective samples.

2. masks directory (204 masks)
Contains the pixel-level ground truth binary masks corresponding to the defective images. The mask files share the exact same names as their corresponding defective images in the images folder. Masks use two intensity levels: 0 for non-defective background areas and 255 for defective regions.

3. MVTec directory
To facilitate seamless integration into existing Artificial Intelligence (AI) anomaly detection frameworks, the high-resolution images were cropped into 256×256-pixel patches. These patches are organized strictly following the standard MVTec Anomaly Detection benchmark directory structure:

- train/good (14,196): Contains only non-defective patches intended for model training.
- test/good (4,969): Contains non-defective patches for evaluation.
- test/defect (687): Contains defective patches for evaluation.
- ground_truth/defect (687): Stores the corresponding 256×256 binary masks for the test/defect patches, maintaining identical filenames.

4. Metadata and Annotation Files

To ensure structured data handling and full reproducibility, the dataset root directory includes four specific auxiliary files:

-Metadata (CSV):
* RAW_FABRID_HighRes_Metadata.csv: Contains structured information (e.g., filename, anonymized fabric origin ID, binary class label, defect area, number of defects) for the original high-resolution images.
* RAW_FABRID_Patches_Metadata.csv: Contains the metadata corresponding to the 256×256 cropped MVTec patches. Crucially, it explicitly maps each patch back to its original high-resolution source image, ensuring full data traceability.

- Annotations (JSON):
* RAW_FABRID_HighRes_COCO.json: Provides precise bounding box and polygon annotations for all defects in the high-resolution images, strictly formatted as COCO (Common Objects in Context) compatible JSON.
* RAW_FABRID_Patches_COCO.json: Provides the corresponding COCO-formatted annotations mapped specifically to the MVTec patch structure.


DATA ACQUISITION
Images were acquired using a custom-built industrial textile inspection machine. The vision system features a Basler line-scan camera capturing at a spatial resolution of 4 pixels per millimeter. The setup utilized dual LED illumination (top and bottom) to properly highlight structural fabric variations and surface defects.

USE CASE
Testing, developing, and benchmarking traditional computer vision or AI-based (Deep Learning) algorithms for anomaly detection, image classification, and semantic segmentation in raw fabrics.
 
KEYWORDS
Artificial Intelligence, Computer Vision, Image Processing, Machine Learning, Textile Industry, Raw Fabrics.

CITATION & MORE INFORMATION
For a comprehensive description of the dataset, quality control procedures, and technical validation, please refer to the associated published data article.