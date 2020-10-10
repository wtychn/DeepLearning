import pandas as pd
import datetime

if __name__ == '__main__':
    df1 = pd.read_csv("IronVelocity.csv")
    df2 = pd.read_csv("IronVelocity2.csv")

    df3 = df1.append(df2)

    df3.to_csv("IronVel.csv", header=False, index=False)
