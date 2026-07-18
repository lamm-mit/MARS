# Paper Metadata Folder README

This folder contains the metadata and count summaries for the literature corpora used to generate the knowledge graphs (KGs).

## Folder contents

### 1. `Full_Paper_metadata.xlsx`
Original metadata file for the PFAS-specific full-text paper corpus.  
The Excel tabs are organized by publisher, since the full papers were collected by publisher.

- Total metadata entries: **4,825**
- Web of Science search results reported in the paper: **4,824**

### 2. `Full_paper_publisher_paper_counts.csv`
Summary file counting the number of full-text paper metadata entries from each publisher tab in `Full_Paper_metadata.xlsx`.

### 3. `Properties abstracts/`
Original property-abstract metadata files.  
The folder contains Excel spreadsheets organized by targeted material property keyword.

- Raw keyword-level abstract records across files: approximately **181k**
- After removing duplicates across property-keyword searches: approximately **160k unique abstracts**

### 4. `Abstracts_paper_counts.csv`
Summary file counting the number of abstract records in each property-keyword Excel file from `Properties abstracts/`.

## Note on the sampled ~60k abstracts
A random sample of **63,222 abstracts** was used for the Material Properties KG experiments due to the computational expense of graph merging, traversal, and retrieval at the full 160k-abstract scale.

