import re
import hashlib
import csv
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from collections import defaultdict


def extract_body_content(file_path):
    """
    Extract the body content from an SGML file.
    """
    with open(file_path, "r", encoding="latin1") as file:
        content = file.read()
    bodies = re.findall(r"<BODY>(.*?)</BODY>", content, re.DOTALL)
    return bodies


def get_k_shingles(text, k):
    """
    Generate k-shingles for the given text.
    """
    text = re.sub(r"\s+", " ", text.strip())  # Normalize whitespace
    return [text[i : i + k] for i in range(len(text) - k + 1)]


def minhash_signature(shingle_set, hash_funcs):
    """
    Compute the minhash signature for a set of shingles.
    """
    signature = []
    for func in hash_funcs:
        min_hash = min([func(shingle) for shingle in shingle_set])
        signature.append(min_hash)
    return signature


def main():
    # Initialize a Spark session
    spark = (
        SparkSession.builder.appName("Reuters21578LSH")
        .master("spark://96.9.210.170:7077")
        .config("spark.hadoop.validateOutputSpecs", "false")
        .config(
            "spark.hadoop.home.dir",
            "C:/Users/Asus/Downloads/spark-3.5.0-bin-hadoop3/spark-3.5.0-bin-hadoop3/bin",
        )
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "14g")
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.5")
        .getOrCreate()
    )
    sc = spark.sparkContext

    # SGML file paths - replace with your file paths
    sgml_files = [
        "Data/reut2-000.sgm",
        "Data/reut2-001.sgm",
        "Data/reut2-002.sgm",
        "Data/reut2-003.sgm",
        "Data/reut2-004.sgm",
        "Data/reut2-005.sgm",
        "Data/reut2-006.sgm",
        "Data/reut2-007.sgm",
        "Data/reut2-008.sgm",
        "Data/reut2-009.sgm",
        "Data/reut2-010.sgm",
        "Data/reut2-011.sgm",
        "Data/reut2-012.sgm",
        "Data/reut2-013.sgm",
        "Data/reut2-014.sgm",
        "Data/reut2-015.sgm",
        "Data/reut2-016.sgm",
        "Data/reut2-017.sgm",
        "Data/reut2-018.sgm",
        "Data/reut2-019.sgm",
        "Data/reut2-020.sgm",
        "Data/reut2-021.sgm",
    ]

    # Specify the value of k for k-shingles
    k = 3  # You can change this value as needed
    H = 50  # Number of hash functions

    # Generate hash functions
    max_shingle_id = 2**32 - 1
    hash_funcs = [
        lambda x: int(hashlib.md5((str(x) + str(i)).encode()).hexdigest(), 16)
        % max_shingle_id
        for i in range(H)
    ]

    # Create an RDD for each file, extract body contents, and then union all RDDs
    rdds = [sc.parallelize(extract_body_content(file)) for file in sgml_files]
    bodies_rdd = sc.union(rdds)

    # Generate k-shingles and map them to document IDs
    doc_shingles = bodies_rdd.flatMap(
        lambda text: get_k_shingles(text, k)
    ).zipWithIndex()

    # Create a shingle-to-document matrix
    shingle_doc_matrix = defaultdict(set)
    for shingle, doc_id in doc_shingles.collect():
        shingle_doc_matrix[shingle].add(doc_id)

    # Save the matrix to a CSV file
    with open("shingle_doc_matrix.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Shingle", "Document IDs"])
        for shingle, doc_ids in shingle_doc_matrix.items():
            writer.writerow([shingle, ",".join(map(str, doc_ids))])

    # Compute MinHash signatures for each document and create an HxN signature matrix
    signatures = defaultdict(lambda: [float("inf")] * H)
    for doc_id, shingle_set in shingle_doc_matrix.items():
        for i, func in enumerate(hash_funcs):
            min_hash = min([func(shingle) for shingle in shingle_set])
            signatures[doc_id][i] = min(min_hash, signatures[doc_id][i])

    # Convert the signature matrix to the desired format and save it
    with open("minhash_signatures.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Document ID"] + [f"Hash Function {i}" for i in range(H)])
        for doc_id, signature in signatures.items():
            writer.writerow([doc_id] + signature)

    # Stop SparkContext
    sc.stop()


if __name__ == "__main__":
    main()
