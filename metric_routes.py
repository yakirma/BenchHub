@app.route('/metrics/<int:metric_id>/download')
def download_metric(metric_id):
    """Download metric code as a .txt file"""
    metric = GlobalMetric.query.get_or_404(metric_id)
    
    # Create response with metric code
    response = make_response(metric.python_code)
    response.headers['Content-Type'] = 'text/plain'
    response.headers['Content-Disposition'] = f'attachment; filename={metric.name}.txt'
    
    return response

@app.route('/metrics/upload', methods=['POST'])
def upload_metric():
    """Upload/update metric from a .txt file"""
    try:
        metric_file = request.files.get('metric_file')
        metric_name = request.form.get('metric_name', '').strip()
        description = request.form.get('description', '').strip()
        is_aggregated = 'is_aggregated' in request.form
        accepts_aggregated_inputs = 'accepts_aggregated_inputs' in request.form
        
        if not metric_file:
            flash('No file uploaded.', 'danger')
            return redirect(url_for('metrics_view'))
        
        # Read Python code from file
        python_code = metric_file.read().decode('utf-8')
        
        # Basic validation
        if not python_code.strip():
            flash('Uploaded file is empty.', 'danger')
            return redirect(url_for('metrics_view'))
        
        # Try to compile to check syntax
        try:
            compile(python_code, '<string>', 'exec')
        except SyntaxError as e:
            flash(f'Python syntax error in uploaded file: {e}', 'danger')
            return redirect(url_for('metrics_view'))
        
        # Check if metric with this name exists
        existing_metric = GlobalMetric.query.filter_by(name=metric_name).first()
        
        if existing_metric:
            # Update existing metric
            existing_metric.python_code = python_code
            existing_metric.description = description or existing_metric.description
            existing_metric.is_aggregated = is_aggregated
            existing_metric.accepts_aggregated_inputs = accepts_aggregated_inputs
            db.session.commit()
            flash(f'Metric "{metric_name}" updated successfully.', 'success')
        else:
            # Create new metric
            if not metric_name:
                flash('Metric name is required for new metrics.', 'danger')
                return redirect(url_for('metrics_view'))
            
            new_metric = GlobalMetric(
                name=metric_name,
                description=description,
                python_code=python_code,
                is_aggregated=is_aggregated,
                accepts_aggregated_inputs=accepts_aggregated_inputs
            )
            db.session.add(new_metric)
            db.session.commit()
            flash(f'Metric "{metric_name}" created successfully.', 'success')
            
    except Exception as e:
        db.session.rollback()
        flash(f'Error uploading metric: {e}', 'danger')
    
    return redirect(url_for('metrics_view'))
