import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer

def relative_sse_metric(y_true, y_pred, eps=1e-12):
    """相对平方和误差：SSE / sum(y_true^2)。eps 防止分母为 0。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum(y_true ** 2)
    if denom < eps:
        denom = eps
    return np.sum((y_pred - y_true) ** 2) / denom


def load_data(excel_path):
    """
    读取 Excel：
      - sheet0: 训练集
      - sheet1: 测试集
      - 每个 sheet 最后一列为 y
      - 里面有一列是药名（中文字符串），要当作类别特征
    """
    train_df = pd.read_excel(excel_path, sheet_name=0)
    test_df = pd.read_excel(excel_path, sheet_name=1)

    # ★ 新增：把所有列名统一转成字符串，避免 int/str 混合
    train_df.columns = train_df.columns.astype(str)
    test_df.columns = test_df.columns.astype(str)

    # 找出 object 类型的列（药名），两张表各有 1 列
    train_cat_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()
    test_cat_cols = test_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # 把药名列统一改名为 "drug"
    if len(train_cat_cols) == 1:
        train_df = train_df.rename(columns={train_cat_cols[0]: "drug"})
    if len(test_cat_cols) == 1:
        test_df = test_df.rename(columns={test_cat_cols[0]: "drug"})

    # 按列位置：前 n-1 列做特征，最后一列是 y
    X_train = train_df.iloc[:, :-1].copy()
    y_train = train_df.iloc[:, -1].astype(float).to_numpy()

    X_test = test_df.iloc[:, :-1].copy()
    # 用“列的位置”对齐：让 X_test 的列名 = X_train 的列名
    # 这样 ColumnTransformer 在 predict 的时候就能找到所有需要的列
    X_test.columns = X_train.columns

    y_test = test_df.iloc[:, -1].astype(float).to_numpy()

    # 打印一下 NaN 情况，方便你自己查看
    print("训练集各列缺失值数量：")
    print(X_train.isna().sum())
    print("训练集 y 缺失值数量：", np.isnan(y_train).sum())

    return X_train, y_train, X_test, y_test


def build_model(X_train):
    """
    用 ColumnTransformer:
      - 数值特征: SimpleImputer(中位数) -> 直接给随机森林
      - 类别特征: SimpleImputer(众数) -> OneHotEncoder
    """
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    print("数值特征列:", numeric_features)
    print("类别特征列:", categorical_features)

    # 数值特征：先用中位数填补缺失
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # 类别特征：先用众数填补缺失，再 OneHot
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,  # 用满 M4 Pro 的多核
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", rf),
        ]
    )

    # 带前缀 regressor__
    param_grid = {
        "regressor__n_estimators": [200, 400],
        "regressor__max_depth": [None, 15],
        "regressor__min_samples_split": [2, 5],
    }

    return model, param_grid


def tune_hyperparameters(model, param_grid, X_train, y_train,
                         cv_results_path="cv_results.csv"):
    """
    5 折交叉验证调参，并把所有超参数组合的结果导出到 CSV。
    """
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(relative_sse_metric, greater_is_better=False),
        cv=cv,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )

    grid.fit(X_train, y_train)

    print(">>>> 最优超参数：", grid.best_params_)
    #print(">>>> 最优交叉验证得分 (neg MSE)：", grid.best_score_)
    print(">>>> 最优交叉验证得分 (relative_sse)：", -grid.best_score_)

    # ---------- 导出所有超参数组合的 CV 结果 ----------
    cv_results = pd.DataFrame(grid.cv_results_)

    # 只挑常用的几列，方便写报告
    cv_results_slim = cv_results[[
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]].copy()

    cv_results_slim["mean_relative_sse"] = -cv_results_slim["mean_test_score"]
    cv_results_slim["std_relative_sse"] = cv_results_slim["std_test_score"]

    cv_results_slim.to_csv(cv_results_path, index=False)
    print(f"已将交叉验证结果保存到: {cv_results_path}")

    best_model = grid.best_estimator_
    return best_model



def evaluate_on_test(model, X_test, y_test,
                     errors_path="test_errors.csv",
                     stats_path="test_error_stats.csv"):
    """
    在测试集上计算并导出：
      - 每个样本的误差
      - 整体平方和相对误差
      - 每个样本相对平方误差的均值和方差
    """
    y_pred = model.predict(X_test)

    diff = y_pred - y_test
    sse = np.sum(diff ** 2)
    sst = np.sum(y_test ** 2)
    relative_sse = sse / sst

    # 每个样本的相对平方误差
    per_sample_relative_sq_error = (diff ** 2) / (y_test ** 2)
    mean_err = per_sample_relative_sq_error.mean()
    var_err = per_sample_relative_sq_error.var(ddof=1)

    print("\n========== 测试集评估结果 ==========")
    print(f"测试集样本数: {len(y_test)}")
    print(f"平方和相对误差 (SSE / sum(y_true^2)): {relative_sse:.6f}")
    print(f"每个样本相对平方误差的均值: {mean_err:.6f}")
    print(f"每个样本相对平方误差的方差: {var_err:.6f}")

    # ------- 把所有样本误差导出到 CSV -------
    df_errors = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "squared_error": diff ** 2,
        "relative_squared_error": per_sample_relative_sq_error,
    })
    df_errors.to_csv(errors_path, index=False)
    print(f"\n已将每个样本的误差保存到: {errors_path}")

    # ------- 把整体统计量导出到 CSV -------
    df_stats = pd.DataFrame({
        "relative_sse": [relative_sse],
        "mean_relative_sq_error": [mean_err],
        "var_relative_sq_error": [var_err],
    })
    df_stats.to_csv(stats_path, index=False)
    print(f"已将整体误差统计保存到: {stats_path}")

    return relative_sse, mean_err, var_err, per_sample_relative_sq_error



def main():
    # 把路径改成你自己的 Excel 路径
    excel_path = r"dataset.xlsx"

    X_train, y_train, X_test, y_test = load_data(excel_path)

    print("训练集形状:", X_train.shape, "  目标长度:", y_train.shape)
    print("测试集形状:", X_test.shape, "  目标长度:", y_test.shape)

    base_model, param_grid = build_model(X_train)

    # 先尝试不做交叉验证，简单 fit 一次，确认数据没问题
    print("\n先做一次简单拟合检查数据...")
    base_model.fit(X_train, y_train)
    print("简单拟合成功，可以开始交叉验证。\n")

    best_model = tune_hyperparameters(base_model, param_grid, X_train, y_train)
    evaluate_on_test(best_model, X_test, y_test)


if __name__ == "__main__":
    main()
