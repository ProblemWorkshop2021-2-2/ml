package org.deeplearning4j.examples.sample;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.callbacks.EvaluationCallback;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class MyEvaluativeListener extends EvaluativeListener {
    public File myObj;
    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        super.iterationDone(model, iteration, epoch);
    }

    @Override
    public void onEpochStart(Model model) {
        super.onEpochStart(model);
    }

    @Override
    public void onEpochEnd(Model model) {
        super.onEpochEnd(model);
        try {
            System.out.println(Paths.get("wyniki.txt"));
            Files.write(Paths.get("wyniki.txt"), ((Double) model.score()).toString().getBytes(), StandardOpenOption.APPEND);
        }catch (IOException e) {
            //exception handling left as an exercise for the reader
        }

    }

    @Override
    protected void invokeListener(Model model) {
        super.invokeListener(model);
    }

    @Override
    protected void evalAtIndex(IEvaluation evaluation, INDArray[] labels, INDArray[] predictions, int index) {
        super.evalAtIndex(evaluation, labels, predictions, index);
    }

    @Override
    public IEvaluation[] getEvaluations() {
        return super.getEvaluations();
    }

    @Override
    public InvocationType getInvocationType() {
        return super.getInvocationType();
    }

    @Override
    public EvaluationCallback getCallback() {
        return super.getCallback();
    }

    @Override
    public void setCallback(EvaluationCallback callback) {
        super.setCallback(callback);
    }

    public MyEvaluativeListener(DataSetIterator iterator, int frequency) {
        super(iterator, frequency);
    }

    public MyEvaluativeListener(DataSetIterator iterator, int frequency, InvocationType type) {
        //ten
        super(iterator, frequency, type);
        try {
            myObj = new File("filename.txt");
            if (myObj.createNewFile()) {
                System.out.println("File created: " + myObj.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    public MyEvaluativeListener(MultiDataSetIterator iterator, int frequency) {
        super(iterator, frequency);
    }

    public MyEvaluativeListener(MultiDataSetIterator iterator, int frequency, InvocationType type) {
        super(iterator, frequency, type);
    }

    public MyEvaluativeListener(DataSetIterator iterator, int frequency, IEvaluation... evaluations) {
        super(iterator, frequency, evaluations);
    }

    public MyEvaluativeListener(DataSetIterator iterator, int frequency, InvocationType type, IEvaluation... evaluations) {
        super(iterator, frequency, type, evaluations);
    }

    public MyEvaluativeListener(MultiDataSetIterator iterator, int frequency, IEvaluation... evaluations) {
        super(iterator, frequency, evaluations);
    }

    public MyEvaluativeListener(MultiDataSetIterator iterator, int frequency, InvocationType type, IEvaluation... evaluations) {
        super(iterator, frequency, type, evaluations);
    }

    public MyEvaluativeListener(DataSet dataSet, int frequency, InvocationType type) {
        super(dataSet, frequency, type);
    }

    public MyEvaluativeListener(MultiDataSet multiDataSet, int frequency, InvocationType type) {
        super(multiDataSet, frequency, type);
    }

    public MyEvaluativeListener(DataSet dataSet, int frequency, InvocationType type, IEvaluation... evaluations) {
        super(dataSet, frequency, type, evaluations);
    }

    public MyEvaluativeListener(MultiDataSet multiDataSet, int frequency, InvocationType type, IEvaluation... evaluations) {
        super(multiDataSet, frequency, type, evaluations);
    }
}
