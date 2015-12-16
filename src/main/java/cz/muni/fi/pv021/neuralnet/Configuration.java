/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.muni.fi.pv021.neuralnet;

import java.io.File;
import java.io.IOException;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.w3c.dom.Node;
import org.w3c.dom.Element;
import org.xml.sax.SAXException;
/**
 *
 * @author Mirek
 */
public class Configuration {
    //private  String settingsFileName = "settings.xml";
    
    private  String trainDataSet = "train.txt";
    private  String testForStopDataset= "testForStop.txt";
    private   String independentTestDataset= "independentTest.txt";

    private String activationFunction= "sigmoid";
    private double sigmoidalParamLambda= 1.0;

    private double tangentialParamA= 1.0;
    private double tangentialParamB= 1.0;
    
    private double learningSpeed = 0.6;
    private int maxTrainingIterations = 1000;
    private double desiredTestError = 0.4;
    
    private int[] architecture;
    
    private double minWeightInit;
    private double maxWeightInit;
    
    public Configuration(String filename) throws ParserConfigurationException, SAXException, IOException{
	
            File inputFile = new File(filename);
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(inputFile);
            doc.getDocumentElement().normalize();

            
            
            if (doc.getElementsByTagName("trainDataSet").getLength() > 0) {
            this.trainDataSet = doc.getElementsByTagName("trainDataSet").item(0).getTextContent();
            }
            if (doc.getElementsByTagName("testForStopDataset").getLength() > 0) {
            this.testForStopDataset 
                    = doc.getElementsByTagName("testForStopDataset").item(0).getTextContent();
            }
            if (doc.getElementsByTagName("independentTestDataset").getLength() > 0) {
            this.independentTestDataset 
                    = doc.getElementsByTagName("independentTestDataset").item(0).getTextContent();
            }
            if (doc.getElementsByTagName("activationFunction").getLength() > 0) {
            this.activationFunction 
                    = doc.getElementsByTagName("activationFunction").item(0).getTextContent();
            }
            if (doc.getElementsByTagName("sigmoidalParamLambda").getLength() > 0) {
            this.sigmoidalParamLambda = Double.parseDouble(doc
                    .getElementsByTagName("sigmoidalParamLambda").item(0).getTextContent());
            }
            if (doc.getElementsByTagName("tangentialParamA").getLength() > 0) {
            this.tangentialParamA = Double.parseDouble(doc
                    .getElementsByTagName("tangentialParamA").item(0).getTextContent());
            }
            if (doc.getElementsByTagName("tangentialParamB").getLength() > 0) {
            this.tangentialParamB = Double.parseDouble(doc
                    .getElementsByTagName("tangentialParamB").item(0).getTextContent());
            }
            if (doc.getElementsByTagName("maxTrainingIterations").getLength() > 0) {
            this.maxTrainingIterations = Integer.parseInt(doc
                    .getElementsByTagName("maxTrainingIterations").item(0).getTextContent());
            }
            if (doc.getElementsByTagName("learningSpeed").getLength() > 0) {
            this.learningSpeed = Double.parseDouble(doc
                    .getElementsByTagName("learningSpeed").item(0).getTextContent());
            }
            if (doc.getElementsByTagName("desiredTestError").getLength() > 0) {
            this.desiredTestError = Double.parseDouble(doc
                    .getElementsByTagName("desiredTestError").item(0).getTextContent());
            }
            if (doc.getElementsByTagName("architecture").getLength() > 0) {
            String[] architectureStr = doc.getElementsByTagName("architecture").item(0).getTextContent().split(",");
            this.architecture = new int[architectureStr.length];
                for (int i = 0; i < architectureStr.length; i++) {
                    this.architecture[i] = Integer.parseInt(architectureStr[i]);
                }
            }
            if (doc.getElementsByTagName("initWeightsInt").getLength() > 0) {
            String[] initWeightsIntStr = doc.getElementsByTagName("initWeightsInt").item(0).getTextContent().split(",");
                if (initWeightsIntStr.length == 2) {
                    this.minWeightInit = Double.parseDouble(initWeightsIntStr[0]);
                    this.maxWeightInit = Double.parseDouble(initWeightsIntStr[1]);
                }
            } 
    }

    /**
     * @return the trainDataSet 
     */
    public String getTrainDataSet() {
        return trainDataSet;
    }

    /**
     * @return the testForStopDataset
     */
    public String getTestForStopDataset() {
        return testForStopDataset;
    }

    /**
     * @return the independentTestDataset
     */
    public String getIndependentTestDataset() {
        return independentTestDataset;
    }

    /**
     * @return the activationFunction
     */
    public String getActivationFunction() {
        return activationFunction;
    }

    /**
     * @return the sigmoidalParamLambda
     */
    public double getSigmoidalParamLambda() {
        return sigmoidalParamLambda;
    }

    /**
     * @return the tangentialParamA
     */
    public double getTangentialParamA() {
        return tangentialParamA;
    }

    /**
     * @return the tangentialParamB
     */
    public double getTangentialParamB() {
        return tangentialParamB;
    }

    /**
     * @return the maxTrainingIterations
     */
    public int getMaxTrainingIterations() {
        return maxTrainingIterations;
    }

    /**
     * @return the training speed
     */
    public double getTrainingSpeed() {
        return this.learningSpeed;
    }

    public double getDesiredTestError() {
        return this.desiredTestError;
    }
    
    public int[] getArchitecture() {
        return this.architecture;
    }
    
    @Override
    public String toString() {
        String confToPrint = new String("");
        confToPrint += "Architecture: " + this.architecture + "\n";
        confToPrint += ("Learning speed: " + this.learningSpeed + ", ActivationFunction: ");
        if (this.activationFunction.equalsIgnoreCase("sigmoid")) {
            confToPrint += "sigmoid (lambda = " + this.sigmoidalParamLambda + ")"; 
        } else if (this.activationFunction.equalsIgnoreCase("tanh")) {
            confToPrint += "tanh (a = " + this.tangentialParamA + ", b = " + this.tangentialParamB + ")"; 
        } else {
            confToPrint += "unknown";
        }
        
        return confToPrint;
    }

    /**
     * @return the minWeightInit
     */
    public double getMinWeightInit() {
        return minWeightInit;
    }

    /**
     * @return the maxWeightInit
     */
    public double getMaxWeightInit() {
        return maxWeightInit;
    }
}
