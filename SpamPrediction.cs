using Microsoft.ML.Data;

namespace DecisionTreeDemo;

public class SpamPrediction
{
    [ColumnName("PredictedLabel")]
    public bool IsSpam { get; set; }
}
