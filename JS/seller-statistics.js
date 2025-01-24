const ctx1 = document.getElementById('myChart').getContext('2d');
const ctx2 = document.getElementById('myChart2').getContext('2d');
const ctx3 = document.getElementById('myChart3').getContext('2d');
const ctx4 = document.getElementById('myChart4').getContext('2d');
const ctx5 = document.getElementById('myChart5').getContext('2d');

// annual sales
new Chart(ctx1, {
type: 'bar',
data: {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'],
    datasets: [{
        label: "revenue",
        backgroundColor: '#5BBCFF',
        data: [250, 100, 300, 890, 656, 777, 102, 676, 900, 453, 200, 675],
    },
    {
        label: "orders",
        backgroundColor: '#FB9AD1',
        data: [200, 140, 222, 454, 989, 677, 240, 456, 700, 343, 230, 675],
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

// product wise sales
new Chart(ctx2, {
    type: 'doughnut',
    data: {
        labels: ['prod1', 'prod2', 'prod3', 'prod5'],
        datasets: [{
            data: [300, 250, 100, 600, 450],
            labels: 'Product sold',
            backgroundColor: ['#CDFADB', '#7BD3EA', '#FFC0D9', '#FF8080', '#80BCBD']
        }]

    },
    options:{
        scales:{
            y:{
                beginAtZero: true
            }
        }
    }
});

// gender distribution graph
new Chart(ctx3, {
    type: 'pie',
    data: {
        labels: ['Male', 'Female', 'Others'],
        datasets: [{
            data: [349, 880, 676],
            backgroundColor: ['#CDFADB', '#7BD3EA', '#FFC0D9']
        }]

    },
    options:{
        scales:{
            y:{
                beginAtZero: true
            }
        }
    }
});


// age distribution graph
new Chart(ctx4, {
    type: 'pie',
    data: {
        labels: ['<25', '25-37', '38-60', '>60'],
        datasets: [{
            data: [122, 343, 454, 222],
            backgroundColor: ['#FF76CE', '#5BBCFF', '#D20062', '#FC6736']
        }]

    },
    options:{
        scales:{
            y:{
                beginAtZero: true
            }
        }
    }
});

new Chart(ctx5, {
    type: 'bar',
    data: {
        labels: ['Hinjwadi', 'Alandi', 'Wagholi', 'Dighi', 'Charoli'],
        datasets: [{
            label: "revenue",
            backgroundColor: '#FF004D',
            data: [250, 100, 300, 890, 656],
        },
        {
            label: "orders",
            backgroundColor: '#90D26D',
            data: [200, 140, 222, 454, 989],
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