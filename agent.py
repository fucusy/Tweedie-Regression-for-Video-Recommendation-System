import torch
import torch.nn as nn
from loss import TweedieLoss
from env import RecommenderEnv
from model import PointWiseModel
from config import device
from config import optimization_epochs


class Agent:
    def __init__(self, num_titles):
        self.num_titles = num_titles

    def make_ranking(self):
        raise NotImplementedError("Subclasses must implement the make_ranking method")

    def understand_agent(self):
        raise NotImplementedError("Subclasses must implement the understand_agent method")

    def collect_user_feedback(self, title_ranking, user_feedback):
        """
        Collect user feedback for a given title ranking
        :param title_ranking: list of title ids that were shown to the user, e.g. [0, 1, 2, ...]
        :param user_feedback: list of tuple of feedback from user, action and watch duration, e.g. [("CLICK", 10), ("SKIP", 0), ("NOT_SEEN", 0), ...]
        """
        raise NotImplementedError("Subclasses must implement the collect_user_feedback method")

    def day_starts(self):
        """
        This method is called at the end of each day.
        The agent can use this method to train their model if the model is designed to updated daily.
        """
        raise NotImplementedError("Subclasses must implement the day_starts method")


class PointWiseModelAgent(Agent):
    def __init__(self, num_titles):
        self.num_titles = num_titles
        self.model = PointWiseModel(num_titles)
        self.valid_title_ids_tensor = []  # each element is a title id
        self.valid_user_feedback_tensor = []  # each element is a user feedback, 1 for CLICK, 0 for SKIP

    def make_ranking(self):
        # Get scores for all titles
        scores = self.model(torch.arange(self.num_titles))
        # Return titles sorted by their scores (highest first)
        return torch.argsort(scores, descending=True)

    def collect_user_feedback(self, title_ids, user_feedback):
        clean_user_feedback = [feedback for feedback, _ in user_feedback]
        for title_id, feedback in zip(title_ids, clean_user_feedback):
            if feedback != "NOT_SEEN":
                self.valid_title_ids_tensor.append(torch.tensor(title_id, dtype=torch.long))
                if feedback == "CLICK":
                    self.valid_user_feedback_tensor.append(torch.tensor(1, dtype=torch.float32))
                else:
                    self.valid_user_feedback_tensor.append(torch.tensor(0, dtype=torch.float32))
            else:
                break

    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("model parameters: ", self.model.parameters())

    def day_starts(self):
        if len(self.valid_title_ids_tensor) > 0:
            self.train()

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        # using Sigmoid Cross Entropy Loss first
        criterion = nn.BCEWithLogitsLoss()

        class UserFeedbackDataset(torch.utils.data.Dataset):
            def __init__(self, valid_title_ids_tensor, valid_user_feedback_tensor):
                self.valid_title_ids_tensor = valid_title_ids_tensor
                self.valid_user_feedback_tensor = valid_user_feedback_tensor

            def __len__(self):
                return len(self.valid_title_ids_tensor)

            def __getitem__(self, idx):
                title_id = self.valid_title_ids_tensor[idx]
                user_feedback = self.valid_user_feedback_tensor[idx]
                return title_id, user_feedback

        dataset = UserFeedbackDataset(self.valid_title_ids_tensor, self.valid_user_feedback_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(optimization_epochs):
            epoch_loss = 0.0
            data_size = 0
            for title_id, user_feedback in dataloader:
                optimizer.zero_grad()
                outputs = self.model(title_id)
                loss = criterion(outputs, user_feedback)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                data_size += len(title_id)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, data points: {data_size}, Loss per data point: {epoch_loss / data_size}")

        print("Training completed")


class WeightedPointWiseModelAgent(Agent):
    def __init__(self, num_titles):
        self.num_titles = num_titles        
        self.model = PointWiseModel(num_titles).to(device)
        self.valid_title_ids_tensor = [] # each element is a title id tensor
        self.valid_user_feedback_tensor = [] # each element is a user feedback, 1 for CLICK, 0 for SKIP
        self.valid_watch_duration_weights_tensor = [] # each element is a watch duration tensor

    def make_ranking(self):
        # Get scores for all titles
        scores = self.model(torch.arange(self.num_titles))
        # Return titles sorted by their scores (highest first)
        return torch.argsort(scores, descending=True)

    def collect_user_feedback(self, title_ids, user_feedback): 
        watch_duration_sum = 0
        valid_size = 0
        for title_id, feedback_tuple in zip(title_ids, user_feedback):
            feedback, watch_duration = feedback_tuple
            watch_duration_sum += watch_duration            
            if feedback != "NOT_SEEN":
                valid_size += 1
                self.valid_title_ids_tensor.append(torch.tensor(title_id, dtype=torch.long, device=device))
                if feedback == "CLICK":
                    self.valid_user_feedback_tensor.append(torch.tensor(1, dtype=torch.float32, device=device))
                else:
                    self.valid_user_feedback_tensor.append(torch.tensor(0, dtype=torch.float32, device=device))                                
            else:
                break
        self.valid_watch_duration_weights_tensor.extend([torch.tensor(watch_duration_sum / valid_size, dtype=torch.float32, device=device)] * valid_size)
    
    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("model parameters: ", self.model.parameters())

    def day_starts(self):
        if len(self.valid_title_ids_tensor) > 0:
            self.train()

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        # using Sigmoid Cross Entropy Loss first
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        class UserFeedbackDataset(torch.utils.data.Dataset):
            def __init__(self, valid_title_ids_tensor, valid_user_feedback_tensor, valid_watch_duration_weights_tensor):
                self.valid_title_ids_tensor = valid_title_ids_tensor
                self.valid_user_feedback_tensor = valid_user_feedback_tensor
                self.valid_watch_duration_weights_tensor = valid_watch_duration_weights_tensor

            def __len__(self):
                return len(self.valid_title_ids_tensor)

            def __getitem__(self, idx):
                title_id = self.valid_title_ids_tensor[idx]
                user_feedback = self.valid_user_feedback_tensor[idx]
                watch_duration = self.valid_watch_duration_weights_tensor[idx]
                return title_id, user_feedback, watch_duration

        dataset = UserFeedbackDataset(self.valid_title_ids_tensor, self.valid_user_feedback_tensor, self.valid_watch_duration_weights_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(optimization_epochs):
            epoch_loss = 0.0
            data_size = 0
            for title_id, user_feedback, watch_duration_weight in dataloader:
                optimizer.zero_grad()
                outputs = self.model(title_id)
                loss = criterion(outputs, user_feedback) * watch_duration_weight
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                data_size += len(title_id)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, data points: {data_size}, Loss per data point: {epoch_loss / data_size}")

        print("Training completed")


class RegressionModelAgent(Agent):
    def __init__(self, num_titles):        
        self.num_titles = num_titles        
        self.model = PointWiseModel(num_titles).to(device)
        self.valid_title_ids_tensor = [] # each element is a title id
        self.valid_watch_duration_tensor = [] # each element is a watch duration

    def make_ranking(self):
        # Get scores for all titles
        scores = self.model(torch.arange(self.num_titles, device=device))
        # Return titles sorted by their scores (highest first)
        return torch.argsort(scores, descending=True)

    def collect_user_feedback(self, title_ids, user_feedback): 
        for title_id, feedback_tuple in zip(title_ids, user_feedback):
            feedback, watch_duration = feedback_tuple

            title_id_tensor = torch.tensor(title_id, dtype=torch.long, device=device)
            watch_duration_tensor = torch.tensor(watch_duration, dtype=torch.float32, device=device)
            if feedback != "NOT_SEEN":
                self.valid_title_ids_tensor.append(title_id_tensor)
                self.valid_watch_duration_tensor.append(watch_duration_tensor)
            else:
                break
    
    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("model parameters: ", self.model.parameters())

    def day_starts(self):
        if len(self.valid_title_ids_tensor) > 0:
            self.train()

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        # using Sigmoid Cross Entropy Loss first
        criterion = nn.MSELoss(reduction='mean')

        class UserFeedbackDataset(torch.utils.data.Dataset):
            def __init__(self, valid_title_ids_tensor, valid_watch_duration_tensor):
                self.valid_title_ids_tensor = valid_title_ids_tensor
                self.valid_watch_duration_tensor = valid_watch_duration_tensor

            def __len__(self):
                return len(self.valid_title_ids_tensor)

            def __getitem__(self, idx):
                title_ids = self.valid_title_ids_tensor[idx]
                watch_duration = self.valid_watch_duration_tensor[idx]
                return title_ids, watch_duration

        dataset = UserFeedbackDataset(self.valid_title_ids_tensor, self.valid_watch_duration_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(optimization_epochs):
            epoch_loss = 0.0
            data_size = 0
            for title_ids, watch_duration in dataloader:
                optimizer.zero_grad()
                outputs = self.model(title_ids)
                loss = criterion(outputs, watch_duration)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                data_size += len(title_ids)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, data points: {data_size}, Loss per data point: {epoch_loss / data_size}")

        print("Training completed")


class TweedieModelAgent(Agent):
    def __init__(self, num_titles):        
        self.num_titles = num_titles        
        self.model = PointWiseModel(num_titles).to(device)
        self.valid_title_ids_tensor = [] # each element is a title id
        self.valid_watch_duration_tensor = [] # each element is a watch duration

    def make_ranking(self):
        # Get scores for all titles
        scores = self.model(torch.arange(self.num_titles, device=device))
        # Return titles sorted by their scores (highest first)
        return torch.argsort(scores, descending=True)

    def collect_user_feedback(self, title_ids, user_feedback): 
        for title_id, feedback_tuple in zip(title_ids, user_feedback):
            feedback, watch_duration = feedback_tuple

            title_id_tensor = torch.tensor(title_id, dtype=torch.long, device=device)
            watch_duration_tensor = torch.tensor(watch_duration, dtype=torch.float32, device=device)
            if feedback != "NOT_SEEN":
                self.valid_title_ids_tensor.append(title_id_tensor)
                self.valid_watch_duration_tensor.append(watch_duration_tensor)
            else:
                break
    
    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("model parameters: ", self.model.parameters())

    def day_starts(self):
        if len(self.valid_title_ids_tensor) > 0:
            self.train()

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        # using Sigmoid Cross Entropy Loss first
        criterion = TweedieLoss(p=1.5)

        class UserFeedbackDataset(torch.utils.data.Dataset):
            def __init__(self, valid_title_ids_tensor, valid_watch_duration_tensor):
                self.valid_title_ids_tensor = valid_title_ids_tensor
                self.valid_watch_duration_tensor = valid_watch_duration_tensor

            def __len__(self):
                return len(self.valid_title_ids_tensor)

            def __getitem__(self, idx):
                title_ids = self.valid_title_ids_tensor[idx]
                watch_duration = self.valid_watch_duration_tensor[idx]
                return title_ids, watch_duration

        dataset = UserFeedbackDataset(self.valid_title_ids_tensor, self.valid_watch_duration_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(optimization_epochs):
            epoch_loss = 0.0
            data_size = 0
            for title_ids, watch_duration in dataloader:
                optimizer.zero_grad()
                outputs = self.model(title_ids)
                loss = criterion(outputs, watch_duration)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                data_size += len(title_ids)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, data points: {data_size}, Loss per data point: {epoch_loss / data_size}")

        print("Training completed")


if __name__ == "__main__":
    title_size = 10
    user_size = 10 * title_size
    run_days = 10

    # if True, the manual edit stage is included on the first few days, and the collected data those days are avaiable to the agent
    include_manual_edit_stage = True 
    edit_stage_days = 3

    total_days = run_days + edit_stage_days if include_manual_edit_stage else run_days 
    
    import time
    start_time = time.time()
    env = RecommenderEnv(num_titles=title_size)
    pointwise_agent = PointWiseModelAgent(num_titles=title_size)
    weighted_pointwise_agent = WeightedPointWiseModelAgent(num_titles=title_size)
    tweedie_agent = TweedieModelAgent(num_titles=title_size)
    regression_agent = RegressionModelAgent(num_titles=title_size)

    agents = [tweedie_agent, regression_agent, weighted_pointwise_agent, pointwise_agent]

    # sort by probability, print the title id and value
    sorted_probabilities = sorted(enumerate(env.click_probabilities), key=lambda x: x[1], reverse=True)
    for title_id, probability in sorted_probabilities:
        print(f"title {title_id}: {round(probability, 2)}")

    reward_when_clicks_maximized, reward_when_watch_duration_maximized = env.get_maximum_expected_reward()
    print("reward_when_clicks_maximized: ", reward_when_clicks_maximized)
    print("reward_when_watch_duration_maximized: ", reward_when_watch_duration_maximized)
    max_reward_clicks = reward_when_clicks_maximized[0]
    max_reward_watch_duration = reward_when_watch_duration_maximized[1]

    logged_data = {}
    for agent in agents:
        logged_data[agent.__class__.__name__] = {"actual_clicks_reward_list": [], "actual_watch_duration_reward_list": [], "max_clicks_reward_list": [], "max_watch_duration_reward_list": []}

    for agent in agents:
        actual_click_reward_total = 0.0
        actual_watch_duration_reward_total = 0.0
        max_clicks_reward_total = 0.0
        max_watch_duration_reward_total = 0.0
        actual_clicks_reward_list = []
        actual_watch_duration_reward_list = []
        max_clicks_reward_list = []
        max_watch_duration_reward_list = []
        print("agent name: ", agent.__class__.__name__)
        for day in range(total_days):
            actual_clicks_the_day = 0.0
            actual_watch_duration_the_day = 0.0
            max_clicks_reward_the_day = 0.0
            max_watch_duration_reward_the_day = 0.0
            manual_edit_stage = include_manual_edit_stage and day < edit_stage_days
            if not manual_edit_stage:
                agent.day_starts()
                
            user_iteration_start_time = time.time()
            if manual_edit_stage:
                title_ranking = env.manual_edit_ranking()
            else:
                title_ranking = agent.make_ranking() #TODO: we will optimize bandits, bandits should offer ranking different for each user based on feedbacks
                
            for user_idx in range(user_size):
                if user_idx % 1000 == 0:
                    print(f"user {user_idx} / {user_size}, time taken: {time.time() - user_iteration_start_time}")
                    user_iteration_start_time = time.time()                
                
                user_feedback = env.get_user_feedback(title_ranking)
                
                click_count = sum(1 for feedback, watch_duration in user_feedback if feedback == "CLICK")
                watch_duration = sum(watch_duration for feedback, watch_duration in user_feedback if feedback == "CLICK")

                actual_clicks_the_day += click_count
                actual_watch_duration_the_day += watch_duration

                max_clicks_reward_the_day += max_reward_clicks
                max_watch_duration_reward_the_day += max_reward_watch_duration

                agent.collect_user_feedback(title_ranking, user_feedback)
                
            actual_click_reward_total += actual_clicks_the_day
            max_clicks_reward_total += max_clicks_reward_the_day
            actual_watch_duration_reward_total += actual_watch_duration_the_day
            max_watch_duration_reward_total += max_watch_duration_reward_the_day
            actual_clicks_reward_list.append(actual_clicks_the_day)
            max_clicks_reward_list.append(max_clicks_reward_the_day)
            actual_watch_duration_reward_list.append(actual_watch_duration_the_day)
            max_watch_duration_reward_list.append(max_watch_duration_reward_the_day)

            logged_data[agent.__class__.__name__]["actual_clicks_reward_list"].append(actual_clicks_the_day)
            logged_data[agent.__class__.__name__]["max_clicks_reward_list"].append(max_clicks_reward_the_day)
            logged_data[agent.__class__.__name__]["actual_watch_duration_reward_list"].append(actual_watch_duration_the_day)
            logged_data[agent.__class__.__name__]["max_watch_duration_reward_list"].append(max_watch_duration_reward_the_day)   
            
            print(f"Day {day}, actual_clicks_reward: {actual_clicks_the_day}, max_clicks_reward: {max_clicks_reward_the_day}, actual_watch_duration_reward: {actual_watch_duration_the_day}, max_watch_duration_reward: {max_watch_duration_reward_the_day}")
        agent.understand_agent()
        print("actual_clicks_reward (rounded): ", round(actual_click_reward_total, 2))
        print("max_clicks_reward (rounded): ", round(max_clicks_reward_total, 2))
        print("actual_watch_duration_reward (rounded): ", round(actual_watch_duration_reward_total, 2))
        print("max_watch_duration_reward (rounded): ", round(max_watch_duration_reward_total, 2))

    # save the logged data to a json file, add timestamp to the filename 
    import pandas as pd
    df = pd.DataFrame(logged_data)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    
    # save it to a folder called logged_data, create the folder if it doesn't exist
    import os
    if not os.path.exists("logged_data"):
        os.makedirs("logged_data")
    import random
    random_seed = random.randint(0, 100000000)
    df.to_json(f"logged_data/logged_data_{timestamp}_{random_seed}.json", index=False)
    
    end_time = time.time()
    print("Time taken: ", end_time - start_time)
