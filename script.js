document.addEventListener('DOMContentLoaded', function () {
    const loadingContainer = document.getElementById('loading');
    const outputSection = document.getElementById('output');
    const accuracyText = document.getElementById('accuracy');
    const confMatrixImg = document.getElementById('confMatrix');
    const treeContainer = document.getElementById('treeContainer');

    // Function to display loading animation
    function displayLoadingSpinner(isLoading) {
        loadingContainer.style.display = isLoading ? 'block' : 'none';
    }

    // Listen for changes in model selection or parameters
    document.querySelectorAll('select, input').forEach(element => {
        element.addEventListener('change', function () {
            displayLoadingSpinner(true); // Show loading animation

            // Gather input values
            const modelType = document.getElementById('modelType').value;
            const criterion = document.getElementById('criterion').value;
            const maxDepth = document.getElementById('maxDepth').value;
            const minSamplesSplit = document.getElementById('minSamplesSplit').value;
            const minSamplesLeaf = document.getElementById('minSamplesLeaf').value;

            // Prepare data for the request
            const requestData = {
                modelType,
                criterion,
                maxDepth,
                minSamplesSplit,
                minSamplesLeaf
            };

            // Fetch request to train the model
            fetch('/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData),
            })
            .then(response => {
                if (!response.ok) throw new Error('Network response was not ok');
                return response.json();
            })
            .then(data => {
                // Display the results
                accuracyText.innerHTML = `Model Accuracy: ${data.accuracy.toFixed(2) * 100}%`;
                confMatrixImg.src = data.confusionMatrixImage;
                confMatrixImg.style.display = 'block';
                if (data.decisionTreeImage) {
                    treeContainer.innerHTML = `<img src="${data.decisionTreeImage}" alt="Decision Tree">`;
                } else {
                    treeContainer.innerHTML = ''; // Clear if not a decision tree model
                }
                outputSection.style.display = 'block'; // Show output section

                // Hide loading animation
                displayLoadingSpinner(false);
            })
            .catch(error => {
                console.error('Error:', error);
                showToast('Error occurred during training. Please try again.', true);
                displayLoadingSpinner(false); // Hide loading animation
            });
        });
    });

    // Toast Notification Function
    function showToast(message, isError = false) {
        const toast = document.createElement('div');
        toast.className = 'toast' + (isError ? ' error' : '');
        toast.innerText = message;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(20px)'; // Slide down effect
            setTimeout(() => document.body.removeChild(toast), 500);
        }, 3000);
    }
});
