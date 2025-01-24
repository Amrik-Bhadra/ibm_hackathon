function renderChart1(months, revenue, orders) {
    const ctx1 = document.getElementById('myChart').getContext('2d');
    new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: months, // Use data from the 'months' array for labels
            datasets: [{
                label: "Revenue",
                backgroundColor: '#5BBCFF',
                data: revenue, // Use data from the 'revenue' array for revenue
            },
            {
                label: "Orders",
                backgroundColor: '#FB9AD1',
                data: orders, // Use data from the 'orders' array for orders
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            },
        }
    });
}

// Function to render the second chart (doughnut chart)
function renderChart2(products, productSold) {
    const ctx2 = document.getElementById('myChart2').getContext('2d');
    new Chart(ctx2, {
        type: 'doughnut',
        data: {
            labels: products, // Use data from the 'products' array for product labels
            datasets: [{
                data: productSold, // Use data from the 'productSold' array for product sold
                backgroundColor: ['#CDFADB', '#7BD3EA', '#FFC0D9', '#FF8080', '#80BCBD']
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}
