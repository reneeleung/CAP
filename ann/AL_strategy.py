import numpy as np
from abc import ABCMeta, abstractmethod
import pandas as pd

class ActiveLearningStrategy(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def select_idx(cls, choices_number: int, seed: int,probs: np.ndarray = None, scores: np.ndarray = None,
                best_paths: np.ndarray = None, masks: np.ndarray = None, entropies: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        probs: [B, L, C]
        scores: [B, L]
        best_path: [B, L]
        masks: [B, L]
        """
        pass


def select_token_limits(index, sentence_length, target_length, seed, scores=None, random=False):
    """
    This function serves to control the total number of tokens selected in AL strategies.
    :param index: ID for each sample
    :param sentence_length: sentence length for each sample
    :param random: random select the samples
    :param score: score for each sample, could be confidence, normalized confidence, which should be sorted by ascending order
    :return: A subset of index which have lowest scores and their sentence length add up to target_length
    """
    # Sample DataFrame (replace this with your actual DataFrame)
    data = {
        "index": index,
        "sentence_length": sentence_length,
        "sent_scores": scores,
    }
    df = pd.DataFrame(data)
    # Initialize variables to track selected indices and their total length
    selected_indices = []
    current_length = 0

    if random is True:
        np.random.seed(seed)
        df = df.sample(frac=1).reset_index(drop=True)
    else:
        if scores is not None:
            # Sort the DataFrame by sentences score
            np.random.seed(seed)
            df = df.sort_values(by="sent_scores", ascending=True)

    # Iterate through rows to select indices
    for _, row in df.iterrows():
        index = row["index"]
        length = row["sentence_length"]
        # Check if adding the current sentence length exceeds the target
        if current_length + length < target_length:
            selected_indices.append(index)
            current_length += length
        # If the total length exceeds the target, stop
        else:
            selected_indices.append(index)
            current_length += length
            break

    return selected_indices


def select_by_cluster(index, sentence_length, target_length, seed, clusterID):
    """
    This function serves to select by clusterID and control the total number of tokens selected in Cluster-based AL strategies.
    :param index: ID for each sample
    :param sentence_length: sentence length for each sample
    :param random: random select the samples
    :param score: score for each sample, could be confidence, normalized confidence, which should be sorted by ascending order
    :return: A subset of index which have lowest scores and their sentence length add up to
    """
    data = {
        "index": index,
        "sentence_length": sentence_length,
        "clusterID": clusterID
    }
    df = pd.DataFrame(data)
    # Initialize variables to track selected indices and their total length
    selected_indices = []
    current_length = 0
    cluster_counts = {cluster_id: 0 for cluster_id in df["clusterID"].unique()}
    np.random.seed(seed)
    df = df.sample(frac=1).reset_index(drop=True)
    np.random.seed(seed)
    df = df.sort_values(by="clusterID", ascending=True).reset_index(drop=True)
    df['subID'] = 0
    for i in range(1,df.shape[0]):
        if df.clusterID[i] == df.clusterID[i-1]:
            df.loc[i, 'subID'] = df.loc[i-1, 'subID'] + 1
    cluster_size_max = max(df.subID)
    cluster_size_min = min(df.subID)

    for subID in range(cluster_size_min,cluster_size_max):
        if current_length<target_length:
            # Iterate through rows to select indices
            df_sub = df[df.subID == subID]
            np.random.seed(seed)
            df_sub = df_sub.sample(frac=1).reset_index(drop=True)
            for _, row in df_sub.iterrows():
                index = row["index"]
                length = row["sentence_length"]
                cluster_id = row["clusterID"]
                # Check if adding the current sentence length exceeds the target
                if current_length + length < target_length:
                    selected_indices.append(index)
                    current_length += length
                    cluster_counts[cluster_id] += 1
                    # If the total length exceeds the target, stop
                else:
                    selected_indices.append(index)
                    current_length += length
                    cluster_counts[cluster_id] += 1
                    break
        else:
            break
    return selected_indices

class RandomSentenceStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int = 20, seed: int = 0, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_paths: np.ndarray = None, masks: np.ndarray = None,entropies: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Random Select Strategy with a limit of the number of sentences given by choices_number. This is the strategy used in initialization, selecting the first 1% sentences.
        """
        if "candidate_number" in kwargs:
            candidate_number = kwargs["candidate_number"]
        else:
            candidate_number = scores.shape[0]
        np.random.seed(seed)
        return np.random.choice(np.arange(candidate_number), size=choices_number, replace=False)


class RandomStrategy(ActiveLearningStrategy):
    @classmethod

    def select_idx(cls, choices_number: int = 5000, seed: int = 0, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_paths: np.ndarray = None, masks: np.ndarray = None,entropies: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Randomly select sentences with a limit of the number of tokens given by choices_number
        """
        if "candidate_number" in kwargs:
            candidate_number = kwargs["candidate_number"]
        else:
            candidate_number = scores.shape[0]
        selected_indices = select_token_limits(index = np.arange(candidate_number).tolist(), sentence_length=kwargs["sentence_length"],
                                               target_length= choices_number, seed=seed, random=True)

        return selected_indices

class LeastConfidenceStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int = 20, seed: int = 0, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_paths: np.ndarray = None, masks: np.ndarray = None, entropies: np.ndarray = None, **kwargs) -> np.ndarray:
        sent_scores = np.prod(scores, axis = 1, where=masks>0)
        selected_indices = select_token_limits(index= np.arange(len(sent_scores)).tolist(),
                                               sentence_length=kwargs["sentence_length"],
                                               target_length=choices_number, seed=seed, random=False,
                                               scores=sent_scores)
        return selected_indices


class NBestSequenceEntropy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int = 20, seed: int = 0, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_paths: np.ndarray = None, masks: np.ndarray = None,entropies: np.ndarray = None, **kwargs) -> np.ndarray:
        # The input is the normalized entropy of N most likely sequence probabilities for each sentence.
        # The higher the entropy is, the more uncertain the model is.
        selected_indices = select_token_limits(index=np.arange(len(entropies)).tolist(),
                                               sentence_length=kwargs["sentence_length"],
                                               target_length=choices_number, seed=seed, random=False,
                                               scores=-entropies)
        return selected_indices


class ClusterBasedStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int = 20, seed: int = 0, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_paths: np.ndarray = None, masks: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Cluster Based Selection Strategy
        """
        if "candidate_number" in kwargs:
            clusterID = kwargs["candidate_number"]
            selected_indices = select_by_cluster(index=np.arange(len(clusterID)).tolist(), sentence_length=kwargs['sentence_length'],
                                                   target_length=choices_number, seed=seed, clusterID=clusterID)
        return selected_indices

class ClusterThenLCStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int = 500, seed: int = 0, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_paths: np.ndarray = None, masks: np.ndarray = None, entropies: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Cluster Then Least Confidence Strategy
        """
        if 'training_losses' in kwargs:
            training_losses = kwargs['training_losses']
            if len(training_losses)>=2:
                loss_changes = [training_losses[i]-training_losses[(i-1)] for i in range(1,len(training_losses))]
                stable_loss_iterations = sum(1 for loss in loss_changes if loss >= -kwargs["change_loss_threshold"] and loss <= 0)
                if stable_loss_iterations >= 1:
                    # there are training loss decrease are below 0.001, it shows the model learn slow, therefore we start to change LC
                    # LC strategy
                    idx = LeastConfidenceStrategy.select_idx(seed=seed, choices_number=choices_number, scores=scores,
                                                             best_paths=best_paths, masks=masks,entropies=entropies,
                                                             candidate_number=kwargs["candidate_number"],sentence_length=kwargs["sentence_length"])
                else:
                    # this means that the training loss changes are always above 0.001, we should still use Cluster
                    # Cluster strategy
                    idx = ClusterBasedStrategy.select_idx(seed=seed, choices_number=choices_number, scores=scores,
                                                          best_paths=best_paths, masks=masks,entropies=entropies,
                                                          candidate_number=kwargs["candidate_number"],sentence_length=kwargs["sentence_length"])
            else:
                # Cluster strategy
                idx = ClusterBasedStrategy.select_idx(seed=seed, choices_number=choices_number, scores=scores,best_paths=best_paths,
                                                      masks=masks,entropies=entropies, candidate_number=kwargs["candidate_number"],
                                                      sentence_length=kwargs["sentence_length"])

        return idx


class ClusterThenNBSEStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int = 500, seed: int = 0, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_paths: np.ndarray = None, masks: np.ndarray = None, entropies: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Cluster Then N Best Sequence Entropy Strategy
        """
        if 'training_losses' in kwargs:
            training_losses = kwargs['training_losses']
            if len(training_losses)>=2:
                loss_changes = [training_losses[i]-training_losses[(i-1)] for i in range(1,len(training_losses))]
                stable_loss_iterations = sum(1 for loss in loss_changes if loss >= -kwargs["change_loss_threshold"] and loss <= 0)
                if stable_loss_iterations >= 1:
                    # there are training loss decrease are below 0.005, it shows the model learn slow, therefore we start to change NBest
                    # NBest strategy
                    idx = NBestSequenceEntropy.select_idx(seed=seed, choices_number=choices_number, scores=scores,
                                                          best_paths=best_paths, masks=masks,entropies=entropies,
                                                         candidate_number=kwargs["candidate_number"],sentence_length=kwargs["sentence_length"])
                else:
                    # Cluster strategy
                    idx = ClusterBasedStrategy.select_idx(seed=seed, choices_number=choices_number, scores=scores,best_paths=best_paths,
                                                        masks=masks, entropies=entropies,
                                                        candidate_number=kwargs["candidate_number"],
                                                        sentence_length=kwargs["sentence_length"])
            else:
                # Cluster strategy
                idx = ClusterBasedStrategy.select_idx(seed=seed, choices_number=choices_number, scores=scores,best_paths=best_paths,
                                                      masks=masks, entropies=entropies,
                                                      candidate_number=kwargs["candidate_number"],
                                                      sentence_length=kwargs["sentence_length"])

        return idx
