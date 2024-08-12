import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.evaluation.Evaluation;
// We use the libraries we have selected
public class DecisionTreeClassificationExample {

    public static void main(String[] args) throws Exception {
        // Upload data from a CSV file
        DataSource source = new DataSource("path/to/your/data.csv");
        Instances data = source.getDataSet();

        // Set the attribute that will be predicted (in this case, the last attribute)
        data.setClassIndex(data.numAttributes() - 1);

        // Create a J48 decision tree model(C4.5)
        J48 model = new J48();

        // Train the model on the data
        model.buildClassifier(data);

        // Сreate a set of test data (for example, select a part of the original set)
        Instances testData = new Instances(data);

        // Evaluate the model
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(model, testData);

        // Output the evaluation results
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
    }
}
// Our training app is ready
