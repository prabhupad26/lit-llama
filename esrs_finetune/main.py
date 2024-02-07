import argparse
import yaml
from sqlalchemy import create_engine, select, MetaData, Table, and_, not_
from sqlalchemy.orm import Session
from srn_data_collector.annotations_utils.data_model import (
    Base,
    BlobLvlAnnotations,
    CompanyDetailsTable,
    ComplianceItems,
    ReportingRequirements,
    RptRequirementsMapping,
    StandardsList,
    ValuesWithRevisions,
    EsrsReqMapping
)
import torch
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm
from llm_utils.prompt_utils.llama2 import build_llama2_prompt
from typing import Dict
import os
import json
import random


IGNORE_INDEX = -1


def get_db_data(session, recommendation_response_path: str, output_raw_json_path: str):
    query = (
        select(
            CompanyDetailsTable.name.label('COMPANY_NAME'),
            ValuesWithRevisions.document_id,
            ValuesWithRevisions.year.label('REVISION_YEAR'),
            RptRequirementsMapping.source,
            ComplianceItems.name.label('COMPLIANCE_ITEM_ID'),
            ValuesWithRevisions.value,
            ValuesWithRevisions.document_ref,
            StandardsList.family,
            BlobLvlAnnotations.blob_text,
            BlobLvlAnnotations.document_ref.label('PAGE_NO'),
            BlobLvlAnnotations.blob_id,
        )
        .select_from(CompanyDetailsTable)
        .join(ValuesWithRevisions, ValuesWithRevisions.company_id == CompanyDetailsTable.id)
        .join(ComplianceItems, ComplianceItems.id == ValuesWithRevisions.compliance_item_id)
        .join(ReportingRequirements, ReportingRequirements.id == ComplianceItems.reporting_requirement_id)
        .join(RptRequirementsMapping, RptRequirementsMapping.reporting_requirement_id == ReportingRequirements.id)
        .join(StandardsList, StandardsList.id == RptRequirementsMapping.standard)
        .join(BlobLvlAnnotations, BlobLvlAnnotations.revision_id == ValuesWithRevisions.id)
        .where(
            and_(
                not_(ValuesWithRevisions.value.in_(["[", "\\\\", "/", ']'])),
                not_(ValuesWithRevisions.document_ref.in_(['', '\\', '/', 'from the previous data'])),
                ValuesWithRevisions.value != '',
                StandardsList.family.like('%esrs%')
            )
        ) #.group_by(CompanyDetailsTable.name, ValuesWithRevisions.document_id, RptRequirementsMapping.source)
        )
    
    esrs_req_mapping_data = session.query(EsrsReqMapping).all()
    esrs_req_mapping_data = {data.section_name: data.text for data in esrs_req_mapping_data}
    

    all_samples = []
    result = session.execute(query)
    for row in tqdm(result, desc="creating samples"):
        samples = {}
        if row.source in esrs_req_mapping_data:
            samples['section_name'] = esrs_req_mapping_data[row.source]
            samples['compliance_item_name'] = row.COMPLIANCE_ITEM_ID
            # samples['output'] = f":::Document_ref {row.PAGE_NO} Blob_id {row.blob_id}::: {row.blob_text}"
            samples['output'] = row.blob_text
            
            recommendation_raw_file = os.path.join(recommendation_response_path, f"{row.document_id}.json")
            
            if os.path.exists(recommendation_raw_file):
                with open(recommendation_raw_file) as f:
                    recommentations_dict = json.load(f)
                
                if row.COMPLIANCE_ITEM_ID in recommentations_dict:
                    for paragraph in recommentations_dict[row.COMPLIANCE_ITEM_ID]:
                        if samples['output'] != paragraph[0]['text']:
                            samples['paragraphs'] = [samples['output'], paragraph[0]['text']]
                            all_samples.append(samples)
                
                #TODO : Remove this hack
                # random_index = random.randint(0, len(paragraphs))
                # paragraphs.insert(random_index, samples['output'])
                # samples['paragraphs'] = paragraphs
                # samples['output_index'] = random_index
                
                # all_samples.append(samples)
    
    with open(os.path.join(output_raw_json_path), 'w') as json_file:
        json.dump(all_samples, json_file, indent=2)


def prepare(
    raw_json_path: str,
    tokenizer_path: str,
    dataset_save_path: str,
    test_split_size: int = 2000,
    max_seq_length: int = 3000,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    
    
    tokenizer = Tokenizer(tokenizer_path)
    
    with open(raw_json_path, "r") as file:
        data = json.load(file)

    # data = data[:10000]
    # Partition the dataset into train and test
    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data, 
        lengths=(train_split_size, test_split_size),
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set_tokenized = []
    for sample in tqdm(train_set):
        paragraphs = sample.pop('paragraphs')
        output = sample.pop('output')
        response = "no"
        for para in paragraphs:
            if output == para:
                response = "yes"
            sample['paragraph'] = para
            sample['response'] = response
            train_set_tokenized.append(prepare_sample(sample, tokenizer, max_length= max_seq_length))
    # train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)]
    torch.save(train_set_tokenized, os.path.join(dataset_save_path, "train.pt"))

    print("Processing test split ...")
    test_set_tokenized = []
    for sample in tqdm(test_set):
        paragraphs = sample.pop('paragraphs')
        output = sample.pop('output')
        response = "no"
        for para in paragraphs:
            if output == para:
                response = "yes"
            sample['paragraph'] = para
            sample['response'] = response
            test_set_tokenized.append(prepare_sample(sample, tokenizer, max_length= max_seq_length))
    # test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set)]
    torch.save(test_set_tokenized, os.path.join(dataset_save_path, "test.pt"))


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):

    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + f"<|assistant|>\nResponse: {example['response']}</s>\n"
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    system_message = """
    <|system|>
    Identify whether the given paragraph is relevant for the given
    Requirement below, your response should only be 'yes' or 'no'
    there should not be any other text apart from the reponse</s>
    """

    # input_message = """
    # <|user|>
    # Requirement: "{section_name}"
    # Sub-Requirement: "{compliance_item_name}"
    # Paragraph:
    # {paragraph}</s>
    # """
    input_message = """
    <|user|>
    Requirement: "{compliance_item_name}"
    Paragraph:
    {paragraph}</s>
    """
    return f"""
    {system_message}
    {input_message.format(
        section_name=example['section_name'],
        compliance_item_name=example['compliance_item_name'],
        paragraph=example['paragraph']
    )}
    """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/cluster/home/repo/my_llm_experiments/lit-gpt/esrs_finetune/main.yaml",
        type=str,
        help="Path to config",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))

    db_file_path = config.pop("db_file_path")
    tokenizer_path = config.pop("tokenizer_path")
    dataset_save_path = config.pop("dataset_save_path")
    recommendation_response_path = config.pop("recommendation_response_path")
    output_raw_json_path = config.pop("output_raw_json_path")
    engine = create_engine(f"sqlite:///{db_file_path}")
    
    # Build all the tables
    metadata = Base.metadata
    Base.metadata.create_all(bind=engine)

    # Creates session
    session = Session(engine)
    # get_db_data(session, recommendation_response_path, output_raw_json_path)
    prepare(output_raw_json_path,tokenizer_path, dataset_save_path)


    session.close()


if __name__ == "__main__":
    main()
