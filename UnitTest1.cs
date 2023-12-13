using System.Diagnostics;
using Microsoft.ML;

namespace DecisionTreeDemo;


public class Tests
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void Test1()
    {
        // Initialize ML Context
        var mlContext = new MLContext();

        // Load Data
        var data = new List<Email>
        {
            new Email { Content = "Buy cheap products now", IsSpam = true },
            new Email { Content = "Meeting at 3 PM", IsSpam = false },
            // Add more data here...
        };

        var trainData = mlContext.Data.LoadFromEnumerable(data);
 
        // Prepare Data and Train Model
        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(Email.Content))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

        var model = pipeline.Fit(trainData);

        // Test Model with a sample email
        var sampleEmail = new Email { Content = "Special discount, buy now!" };
        var predictionEngine = mlContext.Model.CreatePredictionEngine<Email, SpamPrediction>(model);
        var prediction = predictionEngine.Predict(sampleEmail);




        Assert.IsTrue(prediction.IsSpam);
        
        
        Debug.WriteLine($"Email: '{sampleEmail.Content}' is {(prediction.IsSpam ? "spam" : "not spam")}");
        Assert.Pass();
    }
}