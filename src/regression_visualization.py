
import matplotlib.pyplot as plt

def visualize_results(results, model_name):
    """ visualize regression model's results"""
    # show R^2
    fig, ax = plt.subplots(figsize=(10, 5))
    y_test = results['y_test']
    y_pred_test = results['y_pred_test']
    ax.scatter(y_test, y_pred_test, alpha=0.5, s=10)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Delay (minutes)')
    ax.set_ylabel('Predicted Delay (minutes)')
    ax.set_title(f"{model_name}: Predicted vs Actual\nR^2 = {results['test_r2']:.4f}", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # show coefficient/importance
    fig, ax = plt.subplots(figsize=(10, 5))
    features = results['features']
    importance = results['feature_importance'] if 'feature_importance' in results else results['coefficients']

    colors = ["#f9a410", "#ff460e", "#77e977"]
    ax.barh(features, importance, color=colors, alpha=0.8)
    ax.set_xlabel('Importance')
    if model_name == 'Linear Regression':
        ax.set_title(f'{model_name}: Feature Coefficients', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'{model_name}: Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(False)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    for i, imp in enumerate(importance):
        if model_name == 'Linear Regression':
            if imp < 0: 
                ax.text(imp, i, f'  {imp:.1f}', va='center', ha='right')
            else:
                ax.text(imp, i, f'  {imp:.1f}', va='center', ha='left')
        else:
            ax.text(imp, i, f'  {imp:.1f}', va='center', ha='left')
    plt.tight_layout()
    plt.show()