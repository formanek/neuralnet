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
    
    public Trainer(LayeredNeuralNetwork neuralNetwork, Configuration conf) throws IOException {
        
        this.network = neuralNetwork;
        this.trainSet = InputDataCSV.getDataFromFile(conf.getTrainDataSet());
        this.testForStopTest = InputDataCSV.getDataFromFile(conf.getTestForStopDataset());
        this.independentTestSet = InputDataCSV.getDataFromFile(conf.getIndependentTestDataset());
        this.trainingSpeed = conf.getTrainingSpeed();
        this.maxIterations = conf.getMaxTrainingIterations();
        this.desiredTestError = conf.getDesiredTestError();
        this.settingsString = conf.toString();
    }
    
    public void train() throws IOException {
        int iteration = 0;
        double error = 0;
        double testError = 0;
        do {
            error = 0;
            testError = 0;
            int usableTrainDatasetLength = 0;
            int usableTestDatasetLength = 0;
            for (DatasetExample example : this.trainSet) {
                double[] inputs = example.getInputs();
                double[] outputs = example.getOutputs();
                if (inputs.length != this.network.getInputLayer().getNeurons().size() 
                        || outputs.length != this.network.getLayers().get(this.network.getLayers()
                                .size() - 1).getNeurons().size()) {
                    System.err.println("Wrong dataset example length");
                    continue;
                }
                usableTrainDatasetLength++;
                //System.out.println("-----------------------------------------------------------------");
                network.setData(inputs);
                //System.out.println("Weights:" + this.network.weightsToString());
                double[] results = network.getResult();
                //System.out.println("NeuronOuts: " + this.network.outputsToString());
                System.out.println("Input: " + Arrays.toString(inputs));
                System.out.println("\tResults: " + Arrays.toString(results));
                System.out.println("\tRef. Outputs: " + Arrays.toString(outputs));
                //delta precomputation
                List<Neuron> lastLayerNeurons = this.network.getLayers().get(this.network.getLayers().size() - 1).getNeurons();
                double errorTmp = 0;
                //System.out.println("Deltas: " + this.network.deltasToString());
                for (int i = 0; i < lastLayerNeurons.size(); i++) {
                    lastLayerNeurons.get(i).computeLastLayerDelta(outputs[i]);
                    errorTmp += Math.pow(outputs[i] - results[i], 2); 
                            //- lastLayerNeurons.get(i).getFunctionOutput(), 2);
                }
                error += 0.5 * errorTmp;
                
                for(int j = this.network.getLayers().size() - 2; j >= 0; j--) {
                    for(Neuron neuron : this.network.getLayers().get(j).getNeurons()) {
                        neuron.computeDelta();
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
                //System.out.println("WeightChanges: " + this.network.weightChangesToString());
                for (NeuralNetworkLayer layer : this.network.getLayers()) {
                    for (Neuron neuron : layer.getNeurons()) {
                        for (NeuralConnection connection : neuron.getInputs()){ 
                            connection.adaptWeight(this.trainingSpeed);
                            connection.clearWeightChange();
                        }
                    }
                }            
            }

            //System.out.println("new weights:" + this.network.weightsToString());
            //test for stop
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
            error = error / usableTrainDatasetLength;
            testError = testError / usableTestDatasetLength;
            
            System.out.println("....::::Iteration: " + iteration + ": Training error: " 
                    + error + ", Testing error: " 
                    + testError + "::::....desiredTestErr: " + this.desiredTestError);
            iteration++;
            
        } while(iteration < this.maxIterations && testError > this.desiredTestError);
        String report = new String("Neural network training report: ");
        DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
        Date date = new Date();
        report += dateFormat.format(date);
        report += "\n\n" + this.settingsString + "\n\n";
        report += "Iterations: " + iteration + ": Training error: " 
                    + error + ", Testing error: " 
                    + testError + "\n\n\n\n";
        
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
    
}
