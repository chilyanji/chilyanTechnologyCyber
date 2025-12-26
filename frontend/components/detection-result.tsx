interface DetectionResultProps {
  result: {
    classification: "safe" | "suspicious" | "malicious"
    confidence: number
    explanation: string
    risk_level: string
  }
}

const classificationConfig = {
  safe: {
    icon: "✓",
    color: "text-safe",
    bgColor: "bg-safe/10",
    borderColor: "border-safe/30",
    title: "Safe",
  },
  suspicious: {
    icon: "⚠",
    color: "text-suspicious",
    bgColor: "bg-suspicious/10",
    borderColor: "border-suspicious/30",
    title: "Suspicious",
  },
  malicious: {
    icon: "✕",
    color: "text-malicious",
    bgColor: "bg-malicious/10",
    borderColor: "border-malicious/30",
    title: "Malicious",
  },
}

export default function DetectionResult({ result }: DetectionResultProps) {
  const config = classificationConfig[result.classification]

  return (
    <div className={`${config.bgColor} border ${config.borderColor} rounded-xl p-6`}>
      <div className="flex items-start gap-4">
        <span className={`text-4xl ${config.color}`}>{config.icon}</span>
        <div className="flex-1">
          <h3 className={`text-2xl font-bold ${config.color} mb-2`}>{config.title}</h3>
          <p className="text-foreground/70 mb-4">{result.explanation}</p>
          <div className="flex items-center gap-6">
            <div>
              <p className="text-foreground/70 text-sm mb-1">Confidence</p>
              <p className="text-2xl font-bold text-foreground">{(result.confidence * 100).toFixed(0)}%</p>
            </div>
            <div>
              <p className="text-foreground/70 text-sm mb-1">Risk Level</p>
              <p className="text-2xl font-bold text-foreground capitalize">{result.risk_level}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
