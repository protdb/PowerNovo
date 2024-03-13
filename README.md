# PowerNovo: A New Efficient Tool for De Novo Peptide Sequencing #
<img title="a title" alt="Alt text" src="/images/Logo1.png">

## What is this? ##
PowerNovo is a software solution that consists of a set of Transformer and BERT models for peptide sequencing. The solution includes a model for identifying patterns in tandem mass spectra, precursors, fragment ions, and a natural language processing model BERT, which performs the function of evaluating the quality of peptide sequences and helps in reconstructing noisy signals. Additionally, the solution integrates an algorithm for identifying proteins based on peptide sequences, which allows for a complete and comprehensive cycle of spectral data processing.


## Quick Guide ##
### Install PowerNovo ###
    pip install powernovo
    
### Usage ###
Using the tool, is as simple and convenient as possible:

    usage: run.py inputs [-w -o -batch_size -alps -num_contigs -contig_kmers - infer -fasta -use-bert]
    positional arguments:
    inputs              Path to the mgf file or path to the folder containing
                        the list of *.mgf files

**Options:**

    -w Path to the project's working folder. The model weights and ALPS peptide assembler will be automatically loaded into this folder. If the path is not specified, the working folder will be automatically created in the current folder       with the name pwn_work.
    
    -o Path to the folder into which the results of data processing will be extracted. If not specified, the data will be loaded into the pwn_output folder, which will be created in the folder where the input files are located.
    -batch_size Number of simultaneously processed spectra. Default is 16. For GPUs with memory > 2048K, the size can be increased to 32, 64, etc.
    
    -alps Use ALPS peptide assembler during post-processing (default = True)
    
    -num_contigs Number of contigs that will be assembled by the ALPS peptide assembler (default=20)

    -contigs_kmers  kmer lengths that will be used by ALPS when assembling contigs (default [7, 8, 10])
    
    -infer Use an algorithm for matching pepts (contigs) with proteins (default=Тrue) 
    
    -fasta Path to the protein FASTA file that will be used to match peptides (contigs) and proteins. If not specified, FASTA Homo sapiens (9606) will be used.
    
    -use-bert Use the Peptide Bert model to evaluate hypotheses (default True). This option can be disabled if there is insufficient GPU memory with some loss of accuracy in the results.

**You can also run the pipeline directly from code** See example in [Colab notebook](/examples/colab_notebook)

    from powernovo.run import run_inference
    
    if __name__ == '__main__':
        run_inference(
                inputs: str, # input file or folder
                working_folder: str = 'pwn_work', # Path to the project's working folder (see above: -w options).
                output_folder: str = '', # Output folder (see above: -o options)
                batch_size: int = 16,
                use_assembler: bool = True, # (see above: -alps options)
                protein_inference: bool = True, # (see above: -infer options)
                fasta_path: str = '',
                use_bert: bool = True,
                num_contigs: int = 20,
                contigs_kmers: list = [7, 8, 10], # 
                annotated_spectrum: bool = False, # If mfg is annotated, then the prediction results passed 
                                                  #  to the callback function will contain the annotation from mgf
                callback_fn: callable = None  # The function to which the prediction results will be returned
        )
        
        
When launched, the program automatically downloads model weights and ALPS peptide assembler. Download data is located on the Figshare resource 10.6084/m9.figshare.25329586  [Models data](https://figshare.com/s/49d21966f8230445f2a4) 
If necessary, you can download them manually and put them in the working folder.

### Output: ###
The pipeline output results are represented by the following files:
* Table with predicted peptides and their confidence score.
* Fasta files with assembled contigs.
* Table of mapping peptides into proteins and protein groups.
* Table containing solved proteins and their peptide sequences.

Examples of output results can be viewed at [Output results](/examples/pipeline_output)



### Note: ###
To run the ALPS peptide assembler, Java must be installed on the machine. If Java is not installed, try install.

    sudo apt install default-jre

## Pipeline ##
<img title="a title" alt="Alt text" src="/images/pipeline.png">

## Transformer Encode/Decode ##
<img title="a title" alt="Alt text" src="/images/transformer.png">


## Inference peptides with BERT model ##
<img title="a title" alt="Alt text" src="/images/BERT_inference.png">
