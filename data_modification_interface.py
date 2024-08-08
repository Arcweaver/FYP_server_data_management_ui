import streamlit as st
import pandas as pd
import datetime
from datetime import date
import os
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



class NeuralNetworkClassificationModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(NeuralNetworkClassificationModel,self).__init__()
        self.input_layer = nn.Linear(input_dim,128)
        self.h1 = nn.Linear(128,256)
        self.h2 = nn.Linear(256,128)
        self.h3 = nn.Linear(128,64)
        self.output_layer = nn.Linear(64,output_dim)
        self.relu = nn.ReLU()
    
    
    def forward(self,x):
        out =  self.relu(self.input_layer(x))
        out =  self.relu(self.h1(out))
        out =  self.relu(self.h2(out))
        out =  self.relu(self.h3(out))
        out =  self.output_layer(out)
        return out



class voter():
    def __init__(self, input_dim, output_dim, num_epochs, learning_rate, rootFilePath="M:/People/James/test/python/FYP_stuff/"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_epochs = num_epochs
        self.model = NeuralNetworkClassificationModel(input_dim,output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=learning_rate)
        self.rootFilePath = rootFilePath

    def setOutput_dim(self, output_dim):
        self.output_dim = output_dim
        self.model = NeuralNetworkClassificationModel(self.input_dim,self.output_dim)

    def setInput_dim(self, input_dim):
            self.input_dim = input_dim
            self.model = NeuralNetworkClassificationModel(self.input_dim,self.output_dim)


    def get_accuracy_multiclass(self, pred_arr,original_arr):
        if len(pred_arr)!=len(original_arr):
            return False
        pred_arr = pred_arr.numpy()
        original_arr = original_arr.numpy()
        final_pred= []
        for i in range(len(pred_arr)):
            final_pred.append(np.argmax(pred_arr[i]))
        final_pred = np.array(final_pred)
        count = 0
        #here we are doing a simple comparison between the predicted_arr and the original_arr to get the final accuracy
        for i in range(len(original_arr)):
            if final_pred[i] == original_arr[i]:
                count+=1
        return count/len(final_pred)




    def train_network(self,X_train,y_train,X_test,y_test,train_losses,test_losses,plot_acc=False):
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            #forward feed
            output_train = self.model(X_train)
            #calculate the loss
            loss_train = self.criterion(output_train, y_train)
            #backward propagation: calculate gradients
            loss_train.backward()
            #update the weights
            self.optimizer.step() 
            output_test = self.model(X_test)
            loss_test = self.criterion(output_test,y_test)
            train_losses[epoch] = loss_train.item()
            test_losses[epoch] = loss_test.item()
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}")
        
        if plot_acc:
            plt.figure(figsize=(10,10))
            plt.plot(train_losses, label='train loss')
            plt.plot(test_losses, label='test loss')
            plt.legend()
            st.pyplot(plt.gcf())
            predictions_train = []
            predictions_test =  []
            with torch.no_grad():
                predictions_train = self.model(X_train)
                predictions_test = self.model(X_test)
            # Check how the predicted outputs look like and after taking argmax compare with y_train or y_test 
            #predictions_train  
            #y_train,y_test
            train_acc = self.get_accuracy_multiclass(predictions_train,y_train)
            test_acc  = self.get_accuracy_multiclass(predictions_test,y_test)
            st.write(f"Training Accuracy: {round(train_acc*100,3)}")
            st.write(f"Test Accuracy: {round(test_acc*100,3)}")


    def majority_vote(self, input_param):
        today = date.today()
        modelList = []
        unloadableModel = ""

        for i in range(10):
            curday = today - datetime.timedelta(i)
            
            year = curday.year
            month = curday.month
            day = curday.day
            dateString = f"{year:04d}{month:02d}{day:02d}"
            path = self.rootFilePath+"models/"+dateString+'model.pth'
            try:
                model_temp = torch.load(path)
                modelList.append(model_temp)
            except:
                unloadableModel+=path+" "

        majority = None
        for i in range(len(modelList)):
            curModel = modelList[i]
            pred_result = curModel(input_param)[0]
            if majority==None:
                majority = pred_result
            else:
                majority += pred_result

        st.toast("Failed to load the following models: "+unloadableModel)

        return int(majority.argmax())

    def load_network(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
        except:
            st.toast("Can't load network")

    def save_network(self):
        today = date.today()
        year = today.year
        month = today.month
        day = today.day
        dateString = f"{year:04d}{month:02d}{day:02d}"
        path = self.rootFilePath+"models/"+dateString+'model.pth'
        try:
            torch.save(self.model.state_dict(), path)
        except:
            st.toast("Can't save network")

class plotter():

    def plot(data, method, x, y, real_class=None):
        try:
            fig, ax = plt.subplots()
            if method == "plot":
                ax.plot(data[x], data[y]) 
                st.pyplot(fig) 
            elif method == "line":
                st.line_chart(
                    data,
                    x=x,
                    y=y,
                    color=real_class
                    )
            elif method == "scatter":
                st.scatter_chart(
                    data,
                    x=x,
                    y=y,
                    color=real_class
                    )
            elif method == "stem":
                ax.stem(data[x], data[y])  
                st.pyplot(fig)
            elif method == "area":
                st.area_chart(
                    data,
                    x=x,
                    y=y,
                    color=real_class
                    )
            elif method == "bar":
                st.bar_chart(
                    data,
                    x=x,
                    y=y,
                    color=real_class
                    )
        except Exception as e:
            st.toast("Unable to plot the "+method+" graph\nError: "+str(e))

class data_modification_interface():

    def dfToDict(self, df, col1, col2):
        return {a: b for a, b, include in zip(df[col1], df[col2], df['Include']) if include}


    def __init__(self, voter, classFilePath="M:/People/James/test/python/FYP_stuff/data/flower.csv", trainDataPath="put_path_here"):
        self.voter = voter
        self.train_losses = np.zeros(voter.num_epochs)
        self.test_losses  = np.zeros(voter.num_epochs)
        #default path value
        if classFilePath != None:
            self.classFilePath = classFilePath
        if trainDataPath != None:
            self.trainDataPath = trainDataPath

        # get the labels
        try:
            self.flower_mapping_frame = pd.read_csv(self.classFilePath)
        except:
            # some (testing) default data. Please change to back up data in real situation
            flower_label_data = {
                'Flower': ['Setosa', 'Versicolour', 'Virginica', 'Sunflower'],
                'Id': [0, 1, 2, 3],
                'Include': [True, True, True, False]
            }
            self.flower_mapping_frame = pd.DataFrame(flower_label_data)
            self.flower_mapping_frame.to_csv(classFilePath, index=False)
        self.label_mapping = self.dfToDict(self.flower_mapping_frame, 'Id', 'Flower')
        self.output_dim = len(self.label_mapping)

        # get training data
        try:
            self.originalTrainData = pd.read_csv(self.trainDataPath)
        except:
            #have some default value here. Below is temp data
            self.originalTrainData = pd.DataFrame(datasets.load_iris().data, columns=datasets.load_iris().feature_names)
            self.originalTrainData['target'] = datasets.load_iris().target
        # partition training data
        self.X = self.originalTrainData.drop(["target"],axis=1).values
        self.y = self.originalTrainData["target"].values
        self.partitionData()
    
    def partitionData(self): 
        scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30, random_state=42)
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.X_train = torch.FloatTensor(self.X_train)
        self.X_test = torch.FloatTensor(self.X_test)
        self.y_train = torch.LongTensor(self.y_train)
        self.y_test = torch.LongTensor(self.y_test)

    def draw(self):
        st.header("Modify classes and test data here",anchor=False)
        st.divider()
        with st.expander("Class modification"):
            st.subheader("Classes:",anchor=False)
            edited_df = st.data_editor(
                self.flower_mapping_frame,
                column_config={
                    'Id': 'Id',
                    'Flower': 'Flowers',
                    'Include': st.column_config.CheckboxColumn(
                        'Include?',
                        help="Select to include this element",
                        default=False,
                    ),
                },
                num_rows="dynamic"
            )

            if st.button("Confirm Change"):
                edited_df.to_csv(self.classFilePath, index=False)
                self.label_mapping = self.dfToDict(edited_df, 'Id', 'Flower')
                self.output_dim = len(self.label_mapping)
                self.voter.setOutput_dim(self.output_dim)
                st.toast("Change saved")


        with st.expander("Training Data modification"):
            st.subheader("Training Data:",anchor=False)

            st.write("Select displaying axes and display format:")
            self.parameterList = list(self.originalTrainData.columns[:-1])
            self.targetCol = self.originalTrainData[self.originalTrainData.columns[-1]]
            self.displayFormatDict = {"basic plot":"plot", "line chart":"line", "scatter chart":"scatter", "stem chart":"stem graph", "area chart":"area", "bar chart":"bar"}

            col1, col2 = st.columns(2)
            with col1:
                xPresentation = st.selectbox("X axis:",self.parameterList)
            with col2:
                yPresentation = st.selectbox("Y axis: (may not be shown on certain graphs)",self.parameterList)
            
            displayFormat = st.selectbox("Display format:",self.displayFormatDict)
            showDistribution = st.button("Show distribution")

            st.write("Current data distribution:")

            if showDistribution:
                # fig, ax = plt.subplots()
                # try:
                #     getattr(ax, self.displayFormatDict[displayFormat])(xPresentation, yPresentation)
                # except:
                #     getattr(ax, self.displayFormatDict[displayFormat])(xPresentation)
                # st.pyplot(fig)
                
                mapper = self.flower_mapping_frame.set_index('Id')['Flower'].to_dict()
                self.originalTrainData['Flower Name'] = self.originalTrainData['target'].map(mapper)

                graph = plotter.plot(self.originalTrainData, self.displayFormatDict[displayFormat], xPresentation, yPresentation, "Flower Name")
                    



                # # Set the title and axis labels
                # ax.set_title('Histogram of column_name')
                # ax.set_xlabel('Values')
                # ax.set_ylabel('Frequency')
                # # Adjust the layout
                # plt.tight_layout()
                # # Assign the figure and axis objects to variables
                # histogram_fig = fig
                # histogram_ax = ax

            # upload test data and combine csv
            # this part requires further testing
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                newTrainData = pd.read_csv(uploaded_file)
                self.originalTrainData = self.originalTrainData.merge(newTrainData, how='outer')
                self.originalTrainData.to_csv(self.trainDataPath, index=False)
                self.parameterList = list(self.originalTrainData.columns[:-1])
                self.targetCol = self.originalTrainData[self.originalTrainData.columns[-1]]

        
        with st.expander("Model training"):
            st.subheader("Model training:",anchor=False)

            if st.button("Re-partition data"):
                self.partitionData()
                st.toast("Data re-partitioned")

            if st.button("Train Model"):
                st.toast("Network training in progress")
                self.voter.train_network( self.X_train,self.y_train,self.X_test,self.y_test,self.train_losses,self.test_losses,False)
                st.toast("Network training complete")

            if st.button("Save Model"):
                self.voter.save_network()
                st.toast("Network saved")


        


            



    #for real data
    # originalData = pd.DataFrame(originalTrainData.data, columns=originalTrainData.feature_names)
    # originalData['target'] = originalTrainData.target
    # X = originalData.drop(["target"],axis=1).values
    # y = originalData["target"].values
    # scaler = StandardScaler()

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # X_train = torch.FloatTensor(X_train)
    # X_test = torch.FloatTensor(X_test)
    # y_train = torch.LongTensor(y_train)
    # y_test = torch.LongTensor(y_test)



def main():
    input_dim  = 4 
    output_dim = 4
    learning_rate = 0.01
    num_epochs = 1000
    # Below uses the iris data set for testing. 
    # Use the proper data set 
    rootFilePath="M:/People/James/test/python/FYP_stuff/"  # directory for storing networks
    classFilePath="M:/People/James/test/python/FYP_stuff/data/flower.csv" # file of stored labels
    trainDataPath="put_path_here" # file of training data
    majority_voter = voter(input_dim, output_dim, num_epochs, learning_rate, rootFilePath)
    dmi = data_modification_interface(majority_voter, classFilePath, trainDataPath)
    dmi.draw()

main()


