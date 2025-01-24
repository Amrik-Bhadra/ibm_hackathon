const { mysql, express, bodyparser, path, nodemailer, randomstring, session, ejs, fs } = require('./JS/config');
const encoder = bodyparser.urlencoded();

app = express();

app.use(express.static(path.join(__dirname, 'HTML')));
app.set('views', path.join(__dirname, 'HTML'));
app.use("/CSS", express.static("CSS"));
app.use("/JS", express.static("JS"));
app.use("/chart.js-4.4.2", express.static("chart.js-4.4.2"));
app.use("/ASSETS/IMAGES", express.static(path.join(__dirname, "ASSETS", "IMAGES")));
app.use(bodyparser.urlencoded({ extended: true }));
app.set('view engine', 'ejs');


app.use(session({
    secret: 'your-secret-key',
    resave: false,
    saveUninitialized: true
}));

const connection = mysql.createConnection({
    host: "localhost",
    user: "root",
    password: "a1b2c3",
    database: "saarthi"
});

connection.connect(function (error) {
    if (error) throw error;
    else console.log("Connection Successful!");
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, "HTML", "index.html"));
});

app.get('/login', encoder, (req, res) => {
    res.render('forms/login', { errorMsg: null });
});


app.post('/login', encoder, (req, res) => {
    var email = req.body.email;
    var password = req.body.password;

    connection.query("SELECT * from users where email_id = ? AND user_pass = ?", [email, password], (error, fields, results) => {
        if (results.length > 0) {
            req.session.loggedIn = true;
            res.redirect('/');
        }
        else {
            res.render('forms/login', { errorMsg: 'Wrong Email or Password' });
        }
    });
});

app.get('/checkLoginStatus', function (req, res) {
    res.json({ loggedIn: req.session.loggedIn });
});

app.get('/signup', encoder, (req, res) => {
    res.render('forms/signup', { errorMsg: null });
});

app.post('/signup', (req, res) => {
    var email = req.body.email;
    var password = req.body.password;
    var usertype = "Buyer";
    var con_password = req.body.con_password;

    // Query to get the maximum cust_id
    connection.query("SELECT MAX(cust_id) AS max_id FROM users", (maxIdError, maxIdResults) => {
        if (maxIdError) {
            console.error('Error fetching maximum cust_id:', maxIdError);
            res.status(500).send('Internal Server Error');
            return;
        }
        const nextId = maxIdResults[0].max_id + 1;

        if (password === con_password) {
            // Insert the new user with the incremented cust_id
            connection.query("INSERT INTO users (cust_id, email_id, user_pass, user_type) VALUES (?, ?, ?, ?)", [nextId, email, password, usertype], function (error, results, fields) {
                if (error) {
                    console.log(error);
                    res.render('forms/signup', { errorMsg: 'Email already exists' });
                } else {
                    res.redirect('/login');
                }
            });
        } else {
            res.render('forms/signup', { errorMsg: 'Passwords do not match' });
        }
    });
});


let storeOTP = '';

app.get('/forgetPassword', encoder, (req, res) => {
    res.render('forms/forgetPassword', { errorMsg: null });
});

let check_email = '';
app.post('/forgetPassword', encoder, (req, res) => {
    check_email = req.body.email;

    connection.query("SELECT * FROM users WHERE email_id=?", [check_email], function (error, results, fields) {
        if (results.length === 0) {
            res.render('forms/signup', { errorMsg: 'Email not found' });
        } else {
            var otp = randomstring.generate({
                length: 6,
                charset: 'numeric'
            });

            storeOTP = otp;

            var transporter = nodemailer.createTransport({
                service: 'gmail',
                auth: {
                        user: 'srivaths.iyer@gmail.com',
                        pass: 'nhat uenc gfro lgwe'
                }
            });

            var mailOptions = {
                from: 'srivaths.iyer@gmail.com',
                to: check_email,
                subject: 'Password Reset OTP',
                text: 'Your OTP for password reset is: ' + otp
            };

            transporter.sendMail(mailOptions, function (error, info) {
                if (error) {
                    console.log(error);
                    res.status(500).send('Error sending OTP');
                } else {
                    console.log('Email sent: ' + info.response);
                    res.redirect('/verifyotp');
                }
            });
        }
    });
});

app.get('/verifyotp', encoder, (req, res) => {
    res.sendFile(path.join(__dirname, "HTML", "forms", "verify-otp.html"));
});

app.post('/verifyotp', encoder, (req, res) => {
    var input_otp = req.body.otp;

    if (input_otp === storeOTP) {
        res.redirect('/resetpassword');
    } else {
        res.redirect('/forgetpassword');
    }

});

app.get('/resetpassword', encoder, (req, res) => {
    res.render('forms/resetpassword', { errorMsg: null });
});

app.post('/resetpassword', encoder, (req, res) => {
    var password = req.body.password;
    var confirm_password = req.body.confirm_password;

    if (password === confirm_password) {
        connection.query("UPDATE users SET user_pass = ? WHERE email_id = ?", [password, check_email], (error, results, fields) => {
            if (error) {
                res.redirect('/resetpassword');
            } else {
                res.redirect('/login');
            }
        });
    } else {
        res.render('forms/resetpassword', { errorMsg: 'Passwords do not match' });
    }
});

app.get('/category/:category', (req, res) => {
    const category = req.params.category;
    connection.query('SELECT * FROM products WHERE product_category = ?', [category], (error, results) => {
        if (error) throw error;
        res.render('products', { category: category, products: results, successMsg: null });
    });
});

app.post('/add-to-cart', (req, res) => {
    const id = req.body.product_id;
    const weight = req.body.quantity;

    connection.query('SELECT * FROM products WHERE product_id = ?', [id], (error, results) => {
        if (error) {
            throw error;
        }

        if (results.length > 0) {
            const product = results[0];
            connection.query('INSERT INTO cart (product_id, product_name, price, image_code, weight) VALUES (?, ?, ?, ?, ?)',
                [product.product_id, product.product_name, product.price, product.image_code, weight],
                (error, results) => {
                    if (error) {
                        throw error;
                    }
                    connection.query('SELECT * FROM products WHERE product_category = ?', [product.product_category], (error, products) => {
                        if (error) {
                            throw error;
                        }
                        res.render('products', { successMsg: null, products: products });
                    });
                });
        } else {
            res.status(404).send('Product not found!');
        }
    });
});


app.get('/product_page', encoder, (req, res) => {
    const productId = req.query.product_id;

    connection.query('SELECT * FROM products NATURAL JOIN seller WHERE product_id = ?', [productId], (error, results) => {
        if (error) {
            console.error('Error fetching product details:', error);
            res.status(500).send('Internal Server Error');
        } else {
            if (results.length > 0) {
                const product = results[0];
                res.render('product_page', { product: product });
            } else {
                res.status(404).send('Product not found!');
            }
        }
    });
});

app.get('/cart', (req, res) => {
    connection.query('SELECT * FROM cart', (error, results) => {
        if (error) {
            console.error('Error fetching cart items from MySQL:', error);
            res.status(500).send('Internal Server Error');
            return;
        }

        const cartItems = results;

        res.render('cart', { cartItems });
    });
});



app.get('/remove', (req, res) => {
    const productId = req.query.product_id;

    console.log(productId);
    connection.query("DELETE FROM cart WHERE product_id = ?", [productId], (error, results, fields) => {
        if (error) {
            console.error('Removing Process Terminated: ', error);
            return;
        }

        res.redirect('/cart');
    })
});


app.get('/payment', encoder, (req, res) => {
    var grandTotal = req.query.cartTotal;
    res.render('qrcode', { grandTotal });
});

app.get('/seller-login', encoder, (req, res) => {
    res.render('seller/login', { errorMsg: null });
});

app.post('/seller-login', encoder, (req, res) => {
    var email = req.body.email;
    var password = req.body.password;

    connection.query("SELECT * from users where email_id = ? AND user_pass = ?", [email, password], (error, results, fields) => {
        if (results.length > 0) {
            res.redirect('/seller-dashboard');
        } else {
            res.render('seller/login', { errorMsg: 'Wrong Email or Password' });
        }
    });
});

app.get('/seller-signup', encoder, (req, res) => {
    res.render('seller/signup', { errorMsg: null });
});

app.post('/seller-signup', encoder, (req, res) => {
    var email = req.body.email;
    var password = req.body.password;
    var confirmPassword = req.body.con_password;

    if (password === confirmPassword) {
        // Find the maximum seller_id and increment it by 1
        connection.query("SELECT MAX(seller_id) AS max_id FROM seller", function (error, results, fields) {
            if (error) {
                console.error('Error in retrieving max seller_id: ', error);
                res.render('seller/signup', { errorMsg: 'Error in Registration' });
            } else {
                var newSellerId = results[0].max_id + 1;

                // Insert the new seller with incremented seller_id
                connection.query("INSERT INTO seller (seller_id, emailID, password) VALUES (?, ?, ?)", [newSellerId, email, password], function (error, results, fields) {
                    if (error) {
                        console.error('Error in Seller Registration: ', error);
                        res.render('seller/signup', { errorMsg: 'Error in Registration' });
                    } else {
                        // Redirect to the otherDetails page with the new seller_id
                        res.redirect('/otherDetails?seller_id=${newSellerId}');
                    }
                });
            }
        });
    } else {
        res.render('forms/signup', { errorMsg: 'Passwords do not match' });
    }
});

app.get('/otherDetails', (req, res) => {
    var seller_id = req.query.seller_id;
    // console.log(seller_id);
    res.render('seller/otherDetails', { seller_id: seller_id });
});

app.post('/otherDetails', (req, res) => {
    var sellername = req.body.sellername;
    var shopname = req.body.shopname;
    var phoneno = req.body.phoneno;
    var upi = req.body.upi;
    var address = req.body.address;
    var sellerId = req.body.seller_id;

    // Update query to update seller details in the database (excluding latitude and longitude)
    var sql = "UPDATE seller SET name = ?, shopname = ?, phonenumber = ?, upiid = ?, address = ? WHERE seller_id = ?";
    var values = [sellername, shopname, phoneno, upi, address, sellerId];

    // Execute the query
    connection.query(sql, values, (error, results, fields) => {
        if (error) {
            console.error('Error updating seller details:', error);
            res.render('seller/login', { sellerId: sellerId, errorMsg: 'Failed to update details. Please try again.' });
        } else {
            // Construct the query string to pass seller_id to Map_2.html
            var queryString = '?seller_id=${encodeURIComponent(sellerId)}';

            // Redirect to Map_2.html with the query string
            res.redirect('/seller-login');
        }
    });
});



// app.post('/updateLocation', (req, res) => {
//     // Extract all details from request body
//     const { sellername, shopname, phoneno, upi, address, seller_id } = req.query;

//     // Update query to update location and other details in the database
//     const sql = "UPDATE seller SET name = ?, shopname = ?, phoneNumber = ?, upiid = ?, address = ? WHERE seller_id = ?";

//     // Execute the query
//     connection.query(sql, [sellername, shopname, phoneno, upi, address,  seller_id], (error, results, fields) => {
//         if (error) {
//             console.error('Error updating location and details:', error);
//             res.status(500).json({ error: 'Failed to update location and details' });
//         } else {
//             // Optionally, you can send a success response back
//             res.status(200).json({ message: 'Location and details updated successfully' });
//         }
//     });
// });

app.get('/seller-dashboard', encoder, (req, res) => {

    const query = `SELECT
        SUM(order_price) AS total_sales,
        COUNT(DISTINCT cust_id) AS total_customers,
        SUM(order_count) AS total_orders
      FROM
        orders;
    `;

    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching data from MySQL: ' + error.stack);
            res.status(500).send('Internal Server Error');
            return;
        }

        const { total_sales, total_customers, total_orders } = results[0];

        const chartData = {
            totalSales: total_sales,
            totalCustomers: total_customers,
            totalOrders: total_orders,
        };

        res.render('seller/seller-dashboard', { chartData });
    });
});


app.get('/chart-data', (req, res) => {
    const productQuery = `
        SELECT
            p.product_name AS product,
            SUM(sp.order_count) AS quantity_sold
        FROM
            orders sp
        NATURAL JOIN
            products p
        GROUP BY
            p.product_name
        ORDER BY
            quantity_sold DESC;
    `;

    const revenueQuery = `
        SELECT
            MONTH(order_date) AS month,
            SUM(order_price) AS revenue,
            SUM(order_count) AS orders
        FROM
            orders
        GROUP BY
            MONTH(order_date)
        ORDER BY
            MONTH(order_date);
    `;

    connection.query(productQuery, (productError, productResults) => {
        if (productError) {
            console.error('Error fetching product sales data from MySQL:', productError);
            res.status(500).json({ error: 'Internal Server Error' });
            return;
        }

        connection.query(revenueQuery, (revenueError, revenueResults) => {
            if (revenueError) {
                console.error('Error fetching revenue and orders data from MySQL:', revenueError);
                res.status(500).json({ error: 'Internal Server Error' });
                return;
            }

            const products = productResults.map(result => result.product);
            const productSold = productResults.map(result => result.quantity_sold);

            const chartData = {
                products,
                productSold,
                months: revenueResults.map(result => result.month),
                revenue: revenueResults.map(result => result.revenue),
                orders: revenueResults.map(result => result.orders)
            };

            res.json(chartData);
        });
    });
});

app.get('/search', (req, res) => {
    var search_product = req.query.search_product;
    console.log(search_product);
    var query = "SELECT * FROM products WHERE product_name LIKE '%" + search_product + "%'";
    connection.query(query, function (err, rows) {
        if (err) {
            console.log(err);
            res.status(500).json({ error: 'Internal Server Error' });
        }
        else {
            res.render('products', { products: rows, successMsg: null })
        }
    })
});


app.get('/filter', (req, res) => {
    let { min_price, max_price, rating, discount } = req.query;

    // Construct the base query
    let query = "SELECT * FROM products WHERE 1=1";

    // Add filters to the query if they exist
    if (min_price) {
        query += ` AND price >= ${connection.escape(min_price)}`;
    }
    if (max_price) {
        query += ` AND price <= ${connection.escape(max_price)}`;
    }
    if (rating) {
        query += ` AND rating >= ${connection.escape(rating)}`;
    }
    if (discount) {
        // Convert discount from string to integer
        let parsedDiscount = parseInt(discount);

        if (!isNaN(parsedDiscount)) {
            query += ` AND discount >= ${parsedDiscount}`;
        } else {
            // Handle case where discount is not a valid number
            console.log("Invalid discount value:", discount);
            // Optionally, you could respond with an error or adjust the query accordingly
        }
    }

    // Execute the query
    connection.query(query, function (err, rows) {
        if (err) {
            console.log(err);
            res.status(500).json({ error: 'Internal Server Error' });
        } else {
            res.render('products', { products: rows, successMsg: null });
        }
    });
});



app.get('/seller-statistics', encoder, (req, res) => {
    res.sendFile(path.join(__dirname, 'HTML', 'seller', 'seller-statistics.html'));
});

app.get('/cartItemCount', (req, res) => {
    // Query to count items in the cart table
    const query = 'SELECT COUNT(*) AS itemCount FROM cart;';

    // Execute query
    connection.query(query, (error, results) => {
        if (error) {
            console.error('Error fetching cart item count:', error);
            res.status(500).json({ error: 'Internal Server Error' });
            return;
        }
        const itemCount = results[0].itemCount;
        res.json({ itemCount });
    });
});

app.get('/map', (req, res) => {
    res.sendFile(path.join(__dirname, 'HTML', 'Map_2.html'));
});

app.listen(4500);