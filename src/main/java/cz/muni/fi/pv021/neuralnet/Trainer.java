/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.muni.fi.pv021.neuralnet;

import cz.muni.fi.pv021.neuralnet.inputData.InputDataCSV;
import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.sql.Time;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.time.LocalTime;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

/**
 *
 * @author Mirek
 */
public class Trainer {
    private final LayeredNeuralNetwork network;
    private final List<DatasetExample> trainSet;
    private final List<DatasetExample> testForStopTest;
    private final List<DatasetExample> independentTestSet;
    private double trainingSpeed;
    private int maxIterations;
    private double desiredTestError;
    private String settingsString;
    private int recordsBeforeWeightUpdate;
    private int weightUpdatesBeforeTest;
    
    public Trainer(LayeredNeuralNetwork neuralNetwork, Configuration conf) throws IOException {
        
        this.network = neuralNetwork;
        this.trainSet = InputDataCSV.getDataFromFile(conf.getTrainDataSet());
        this.testForStopTest = InputDataCSV.getDataFromFile(conf.getTestForStopDataset());
        this.independentTestSet = InputDataCSV.getDataFromFile(conf.getIndependentTestDataset());
        this.trainingSpeed = conf.getTrainingSpeed();
        this.maxIterations = conf.getMaxTrainingIterations();
        this.desiredTestError = conf.getDesiredTestError();
        this.settingsString = conf.toString();
        this.recordsBeforeWeightUpdate = conf.getRecordsBeforeWeightUpdate();
        this.weightUpdatesBeforeTest = conf.getWeightsUpdatesBeforeTest();
    }
    
    public void train() throws IOException {
        int iteration = 0;
        int trainRecordsUsed = 0;
        double trainError = 0;
        int usableTrainDatasetLength = 0;
        int usableTestDatasetLength = 0;
        boolean enoughTraining = false;
        do {

            trainRecordsUsed = 0;
            for (DatasetExample example : this.trainSet) {
                double[] inputs = example.getInputs();
                double[] outputs = example.getOutputs();
                if (inputs.length != this.network.getInputLayer().getNeurons().size() 
                        || outputs.length != this.network.getLayers().get(this.network.getLayers()
                                .size() - 1).getNeurons().size()) {
                    System.err.println("Wrong dataset example length");
                    continue;
                }
                
                //System.out.println("-----------------------------------------------------------------");
                network.setData(inputs);
                //System.out.println("Weights:" + this.network.weightsToString());
                double[] results = network.getResult();
                //System.out.println("NeuronOuts: " + this.network.outputsToString());
                /*System.out.println("Input: " + Arrays.toString(inputs));
                System.out.println("\tResults: " + Arrays.toString(results));
                System.out.println("\tRef. Outputs: " + Arrays.toString(outputs));*/
                //delta precomputation
                List<Neuron> lastLayerNeurons = this.network.getLayers().get(this.network.getLayers().size() - 1).getNeurons();
                double errorTmp = 0;
                //System.out.println("Deltas: " + this.network.deltasToString());
                for (int i = 0; i < lastLayerNeurons.size(); i++) {
                    lastLayerNeurons.get(i).computeLastLayerDelta(outputs[i]);
                    //errorTmp += Math.pow(outputs[i] - results[i], 2); 
                }
                if (this.indexOfMax(outputs) != this.indexOfMax(results)) {
                    errorTmp++;
                }
                trainError += /*0.5 * */errorTmp;
                usableTrainDatasetLength++;
                for(int j = this.network.getLayers().size() - 2; j >= 0; j--) {
                    for(Neuron neuron : this.network.getLayers().get(j).getNeurons()) {
                        neuron.computeDelta();
                    }
                }

                trainRecordsUsed++;
                if (trainRecordsUsed % this.recordsBeforeWeightUpdate == 0 
                        || trainRecordsUsed == trainSet.size()-1) {
                    //System.out.println("WeightChanges: " + this.network.weightChangesToString());
                    for (NeuralNetworkLayer layer : this.network.getLayers()) {
                        for (Neuron neuron : layer.getNeurons()) {
                            for (NeuralConnection connection : neuron.getInputs()){ 
                                connection.adaptWeight(this.trainingSpeed);
                                connection.clearWeightChange();
                            }
                        }
                    }
                    if ((trainRecordsUsed / this.recordsBeforeWeightUpdate) 
                            % this.weightUpdatesBeforeTest == 0 
                            || trainRecordsUsed == trainSet.size()-1) {
                        
                        double testError = 0;
                        //System.out.println("new weights:" + this.network.weightsToString());
                        //test for stop
                        for (DatasetExample testExample : this.testForStopTest) {
                            double[] testInputs = testExample.getInputs();
                            double[] testOutputs = testExample.getOutputs();
                            if (testInputs.length != this.network.getInputLayer().getNeurons().size() 
                                    || testOutputs.length != this.network.getLayers().get(this.network.getLayers()
                                            .size() - 1).getNeurons().size()) {
                                System.err.println("Wrong dataset example length");
                                continue;
                            }
                            
                            double[] testResults = network.getResult();
                            this.network.getLayers().get(this.network.getLayers().size() - 1).getNeurons();
                            double testErrorTmp = 0;
                            //System.out.println("Deltas: " + this.network.deltasToString());
                            //for (int i = 0; i < lastLayerNeurons.size(); i++) {
                            //    testErrorTmp += Math.pow(testOutputs[i] - testResults[i], 2);
                            //}
                            if (this.indexOfMax(testOutputs) != this.indexOfMax(testResults)) {
                                testErrorTmp++;
                            }
                            testError += /*0.5 **/ testErrorTmp;
                            usableTestDatasetLength++;
                        }
                        System.out.println(trainError + " / " +  testError + " / " + usableTrainDatasetLength + " / " + usableTestDatasetLength);
                        trainError = trainError / usableTrainDatasetLength;
                        testError = testError / usableTestDatasetLength;
                        if (testError < this.desiredTestError) {
                            enoughTraining = true;
                        }
                        System.out.println("....::::Iteration: " + iteration + " (used " + trainRecordsUsed 
                                + " records) Training error: " 
                                + trainError + ", Testing error: " 
                                + testError + "::::....desiredTestErr: " + this.desiredTestError);
                        
                        trainError = 0;
                        testError = 0;
                        usableTrainDatasetLength = 0;
                        usableTestDatasetLength = 0;
                    }
                }

                //System.out.println("Deltas: " + this.network.deltasToString());
                //weights adaptation
                for (NeuralNetworkLayer layer : this.network.getLayers()) {
                    for (Neuron neuron : layer.getNeurons()) {
                        for (NeuralConnection connection : neuron.getInputs()){ 
                            connection.incrementWeightChange(this.trainingSpeed);
                        }
                    }
                }
            }


            iteration++;
            
        } while(iteration < this.maxIterations && !enoughTraining);
        String report = new String("Neural network training report: ");
        DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
        Date date = new Date();
        report += dateFormat.format(date);
        report += "\n\n" + this.settingsString + "\n\n";
        report += "Iterations: " + iteration + ": Training error: " 
                    + trainError + ", Testing error: "; 
                    //+ testError + "\n\n\n\n";
        
        this.writeReport(report);
        
    }
    
    public void test() throws IOException {
        double testError = 0;
        int usableTestDatasetLength = 0;
        for (DatasetExample example : this.testForStopTest) {
        double[] inputs = example.getInputs();
        double[] outputs = example.getOutputs();
        if (inputs.length != this.network.getInputLayer().getNeurons().size() 
                || outputs.length != this.network.getLayers().get(this.network.getLayers()
                        .size() - 1).getNeurons().size()) {
            System.err.println("Wrong dataset example length");
            continue;
        }
        usableTestDatasetLength++;
        double[] results = network.getResult();
        List<Neuron> lastLayerNeurons = this.network.getLayers().get(this.network.getLayers().size() - 1).getNeurons();
        double errorTmp = 0;
        //System.out.println("Deltas: " + this.network.deltasToString());
            for (int i = 0; i < lastLayerNeurons.size(); i++) {
                errorTmp += Math.pow(outputs[i] - results[i], 2);
            }
            testError += 0.5 * errorTmp;
        }
        
                testError = testError / usableTestDatasetLength;

        System.out.println("....::::Testing: Testing error: " 
                + testError + "::::....");
        String report = new String("Neural network testing report: ");
        DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
        Date date = new Date();
        report += dateFormat.format(date);
        report += "\n\n" + this.settingsString + "\n\n";
        report += "Testing error: " 
                    + testError + "\n\n\n\n";
        
        this.writeReport(report);
        
    }
    
    private void writeReport(String report) throws IOException {
        File theDir = new File("log");
        if (!theDir.exists()) {
            try {
                theDir.mkdir();
            } 
            catch (SecurityException e) {
                System.err.println("Cannot write log - " + e);
            }
        }
        Path path = FileSystems.getDefault().getPath("", "log/log.txt");
        if (Files.exists(path)) {
            Files.write(path, report.getBytes(), StandardOpenOption.APPEND);
        } else {
            Files.write(path, report.getBytes(), StandardOpenOption.CREATE);
            }
    }
    
    private int indexOfMax(double[] array) {
        double max = -100.0;
        int maxIndex= 0;
        //System.out.println("Pole: " + Arrays.toString(array));
        for(int i =0; i < array.length; i++) {

            if(array[i] > max) {
                maxIndex = i;
                max = array[i];
                //System.out.println("Zmena indexu na " + maxIndex);
            }
        }
        return maxIndex;
        
    }
}
