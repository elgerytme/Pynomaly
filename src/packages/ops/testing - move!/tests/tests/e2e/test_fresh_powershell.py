import sys

sys.path.insert(0, "C:\\Users\\andre\\Pynomaly\\src")

print("FRESH POWERSHELL Test")
print("Python:", sys.version.split()[0])

try:
    print("SUCCESS - Package import")

    from pynomaly.presentation.sdk.models import AnomalyScore

    score = AnomalyScore(value=0.95, confidence=0.88)
    print("SUCCESS - Model creation:", score.value)

    print("SUCCESS - Client import")

    print("FRESH POWERSHELL - Complete success!")

except Exception as e:
    print("FAILED:", str(e))
