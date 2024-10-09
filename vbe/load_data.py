import json
import pandas as pd
import numpy as np
from typing import List, Tuple, Set

def parse_votes(votes: str) -> List[int]:
    """
    Parse a string of votes into a list of integers.
    
    Args:
        votes (str): A string of votes in the format '[vote1,vote2,...]'.
    
    Returns:
        List[int]: A list of integer votes.
    """
    votes = votes[1:-1]  # Remove the brackets
    votes_list = votes.split(',')  # Split the votes
    return [int(vote) for vote in votes_list]  # Convert to integers

def process_ballot(ballot_text: str, voter_id: str, all_project_ids: Set[str], idx: int) -> pd.DataFrame:
    """
    Process a single ballot text and return the corresponding vote allocation DataFrame.
    
    Args:
        ballot_text (str): The ballot text.
        voter_id (str): The voter ID.
        all_project_ids (Set[str]): A set of all project IDs.
        idx (int): The index of the current ballot.
    
    Returns:
        pd.DataFrame: A DataFrame with the vote allocations.
    """
    ballot_text = ballot_text[1:-1]  # Remove brackets
    ballot = ballot_text.split(',')  # Split the text into individual allocations

    vote_allocation = {}
    new_data = pd.DataFrame(columns=['voterId', 'projectId', 'amount'])
    
    # Process JSON format for each voting power allocation in the ballot
    for allocation_text in ballot:
        allocation_text = allocation_text.replace("{", "").replace("}", "").replace('"', '')
        key, value = allocation_text.split(':')
        vote_allocation[key.strip()] = value.strip()
        
        # If a projectId exists, add it to the new voting allocation DataFrame
        if 'projectId' in vote_allocation:
            project_id = vote_allocation['projectId']
            all_project_ids.add(project_id)
            try:
                amount = float(vote_allocation['amount'])
            except ValueError:
                print(f"Amount is not a float: {vote_allocation['amount']}")
                amount = 0.0

            new_entry = pd.DataFrame({
                'voterId': [voter_id],
                'projectId': [project_id],
                'amount': [amount]
            })
            
            if new_data.empty:
                new_data = new_entry
                
            new_data = pd.concat([new_data, new_entry], ignore_index=False)
    
    return new_data

def process_vote_df(data: pd.DataFrame) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Load data from a DataFrame and process the votes.
    
    Args:
        data (pd.DataFrame): The input data.
    
    Returns:
        Tuple[pd.DataFrame, Set[str]]: A tuple containing the vote allocation DataFrame and a set of all project IDs.
    """
    all_project_ids = set()
    vote_allocation_df = pd.DataFrame(columns=['voterId', 'projectId', 'amount'])
    
    # Iterate through votes array for each voter and process the ballot
    for idx, ballot_text in enumerate(data.iloc[:,7]): # 7 is the index of the 'votes' column
        voter_id = data.iloc[:,0][idx] # 0 is the index of the 'voterId' column
        new_data = process_ballot(ballot_text, voter_id, all_project_ids, idx)
        
        if vote_allocation_df.empty:
            vote_allocation_df = new_data
        else:
            vote_allocation_df = pd.concat([vote_allocation_df, new_data], ignore_index=True)
    
    return vote_allocation_df.reset_index(drop=True), all_project_ids

def featurize_df(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Featurize the vote allocation DataFrame.
    
    Args:
        df (pd.DataFrame): The vote allocation DataFrame.
    
    Returns:
        Tuple[np.ndarray, pd.DataFrame]: A tuple containing the feature vectors and the pivoted DataFrame.
    """
    pivot_df = df.pivot(index='voterId', columns='projectId', values='amount')
    feature_vectors = pivot_df.values
    return feature_vectors, pivot_df

def load_data_rpgf(path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load and process RPGF data from a CSV file.
    
    Args:
        path (str): The path to the CSV file.
    
    Returns:
        Tuple[np.ndarray, pd.DataFrame]: A tuple containing the feature vectors and the pivoted DataFrame.
    """
    data = pd.read_csv(path)
    vote_allocation_df, all_project_ids = process_vote_df(data)
    
    print("\nNumber of total projects:", len(all_project_ids))
    print("Number of total projects:", len(vote_allocation_df['voterId']), "\n")
    
    vote_allocation_df = vote_allocation_df.drop_duplicates()
    vote_allocation_df = vote_allocation_df.groupby(['voterId', 'projectId'], as_index=False).sum()
    
    feature_vectors, pivot_df = featurize_df(vote_allocation_df)
    
    return feature_vectors, pivot_df
