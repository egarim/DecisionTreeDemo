using Microsoft.ML.Data;

namespace DecisionTreeDemo;

public class Email
{
    [LoadColumn(0)]
    public string Content { get; set; }

    [LoadColumn(1), ColumnName("Label")]
    public bool IsSpam { get; set; }
}
