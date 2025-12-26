import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
meta = pd.read_csv("data/metaData.csv")

train = train.merge(meta, on="lead_code", how="left")
test = test.merge(meta, on="lead_code", how="left")

wa = pd.read_csv("data/whatsapp_activity.csv")

wa_feat = wa.groupby("lead_code").agg(
    wa_sent=("sent_at", "count"),
    wa_read=("read_at", lambda x: x.notnull().sum())
).reset_index()

train = train.merge(wa_feat, on="lead_code", how="left")
test = test.merge(wa_feat, on="lead_code", how="left")


human = pd.read_csv("data/call_placed.csv")
bot = pd.read_csv("data/teleco_call_back.csv")

human_feat = human.groupby("lead_code").agg(
    human_calls=("start_time", "count"),
    human_answered=("disposition", lambda x: (x=="ANSWERED").sum())
).reset_index()

bot_feat = bot.groupby("lead_code").agg(
    bot_calls=("start_time", "count"),
    bot_answered=("disposition", lambda x: (x=="ANSWERED").sum())
).reset_index()

train = train.merge(human_feat, on="lead_code", how="left")
train = train.merge(bot_feat, on="lead_code", how="left")

test = test.merge(human_feat, on="lead_code", how="left")
test = test.merge(bot_feat, on="lead_code", how="left")


field = pd.read_csv("data/mobile_app_data.csv")

field_feat = field.groupby("lead_code").agg(
    visits=("visit_date", "count"),
    met_customer=("result", lambda x: (x=="MET_CUSTOMER").sum())
).reset_index()

train = train.merge(field_feat, on="lead_code", how="left")
test = test.merge(field_feat, on="lead_code", how="left")


sms = pd.read_csv("data/AI_sms_callback.csv")

sms_feat = sms.groupby("lead_code").agg(
    sms_sent=("status", "count")
).reset_index()

train = train.merge(sms_feat, on="lead_code", how="left")
test = test.merge(sms_feat, on="lead_code", how="left")

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

from sklearn.preprocessing import LabelEncoder

cols = ["suggested_action", "dpd_bucket", "state"]

for col in cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))


from sklearn.linear_model import LinearRegression

X = train.drop(columns=["id", "TARGET","lead_code"])
y = train["TARGET"]


model = LinearRegression()
model.fit(X, y)

print("Model training completed")
X_test = test.drop(columns=["id", "lead_code"])

test_probs = model.predict(X_test)

# Clip values between 0 and 1 (VERY IMPORTANT)
test_probs = test_probs.clip(0, 1)

submission = pd.DataFrame({
    "id": test["id"],
    "TARGET": test_probs
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully")
